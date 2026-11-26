# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from pydantic import BaseModel

from methods.prompts.literature_review_agent import (
    literature_review_agent_writter_prompt,
)
from utils.llm import is_gemini_model, web_search_tools
from utils.llm_backend_utils import call_llm_with_contents
from utils.llm_parsing import parse_latex
from utils.model_defaults import DEFAULT_FAST_MODEL, DEFAULT_WRITER_MODEL
from utils.scholar_utils import keyword_search, title_candidates, title_search

SEARCH_BACKENDS = ("auto", "google", "s2")
# LLM-assisted recovery attempts per run for candidates not found by title.
DEFAULT_MAX_CITATION_RECOVERIES = int(os.getenv("PAPER_MAX_CITATION_RECOVERIES", "40"))


# Minimum LLM relevance score (0-3) for exploration results to be kept.
MIN_RELEVANCE_SCORE = int(os.getenv("PAPER_MIN_RELEVANCE_SCORE", "2"))

_QUERY_FILLER = re.compile(
    r"\b(find|identify|search(?: for)?|look for|locate|highly[- ]cited|influential|"
    r"seminal|foundational|landmark|key|recent|published|research|papers?|works?|"
    r"studies|articles?|establishing|demonstrating|showing|introducing|on the|"
    r"real[- ]world impact of|impact of|technical report|original|the|of|for|in|on|"
    r"and|a|an|that|which|about)\b",
    re.IGNORECASE,
)


def clean_search_query(query: str, max_words: int = 8) -> str:
    """Strips instruction/filler words that make keyword search drift and caps
    the length (long queries match loosely in full-text search engines)."""
    text = re.sub(r"[\"'‘’“”()\[\]{}]", " ", query or "")
    text = _QUERY_FILLER.sub(" ", text)
    words = [w for w in re.split(r"\s+", text) if w and w not in {"-", ":", ","}]
    return " ".join(words[:max_words]).strip(" ,.:;-")


_GENERIC_NAME_WORDS = {
    "model", "models", "dataset", "datasets", "benchmark", "metric", "metrics",
    "optimizer", "method", "methods", "framework", "network", "networks", "module",
    "modules", "variant", "version", "architecture", "algorithm", "approach",
}


def title_lookup_variants(queries: List[str]) -> List[str]:
    """Title-lookup strings for an original paper: each query plus a variant
    without generic words, since title matching requires every word
    ("segment anything model" misses the paper titled "Segment Anything")."""
    variants = []
    for q in queries:
        variants.append(q)
        core = " ".join(w for w in q.split() if w.lower() not in _GENERIC_NAME_WORDS)
        if core and len(core.split()) >= 1 and core.lower() != q.lower():
            variants.append(core)
    return list(dict.fromkeys(v for v in variants if v))


def parse_relevance_scores(parsed) -> List[Dict]:
    """Normalizes the LLM's relevance output to [{"index", "score", "reason"}].

    Accepts {"scores": [...]}, a bare list, or {"0": 3, "1": 0} mappings.
    """
    if isinstance(parsed, dict) and "scores" in parsed:
        parsed = parsed["scores"]
    items = []
    if isinstance(parsed, dict):
        parsed = [{"index": k, "score": v} for k, v in parsed.items()]
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(str(item.get("index")).strip("[] "))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        items.append({"index": index, "score": score, "reason": str(item.get("reason") or "")})
    return items


class CandidatePaper(BaseModel):
    title: str
    year: int
    reason: str


class DiscoveryResult(BaseModel):
    section_name: str
    candidates: List[CandidatePaper]


class PaperData(BaseModel):
    citation_key: str
    title: str
    authors: List[str]
    venue: str
    year: int
    abstract: str
    citation_count: int
    found_in_section: str
    reason: str
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publication_date: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None


class WriterOutput(BaseModel):
    sections_written: List[str]
    latex_code: str


class HybridLiteratureAgent:
    def __init__(
        self,
        idea_path: str,
        experimental_log_path: str,
        latex_template_path: str,
        conference_guidelines_path: str,
        output_dir: str,
        model_name: str = DEFAULT_WRITER_MODEL,
        max_workers: int = 5,
        fast_model_name: str = DEFAULT_FAST_MODEL,
        search_backend: str = "auto",
        max_citation_recoveries: int = DEFAULT_MAX_CITATION_RECOVERIES,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model_name = model_name
        # Number of background workers verifying candidates (lookups are
        # additionally rate-limited per source in utils.scholar_utils).
        self.max_workers = max(1, max_workers)
        # Max LLM-assisted recovery attempts for candidates that could not be
        # found by title (0 disables recovery).
        self.max_citation_recoveries = max(0, max_citation_recoveries)
        self._recoveries_used = 0
        self._inflight_titles = set()
        self.verification_log = {"recovered": [], "rejected": []}
        self.context_paths = {
            "idea.md": idea_path,
            "experimental_log.md": experimental_log_path,
            "template.tex": latex_template_path,
            "conference_guidelines.md": conference_guidelines_path,
        }
        self.context_files = self._load_context_files()
        self.fast_model_name = fast_model_name
        self.search_backend = self._resolve_search_backend(search_backend)

        self.registry_lock = threading.Lock()
        self.print_lock = threading.Lock()

    def _resolve_search_backend(self, backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend not in SEARCH_BACKENDS:
            raise ValueError(
                f"Unknown search backend '{backend}'. Choose from {SEARCH_BACKENDS}."
            )
        fast_is_gemini = is_gemini_model(self.fast_model_name)
        if backend == "auto":
            return "google" if fast_is_gemini else "s2"
        if backend == "google" and not fast_is_gemini:
            print(
                f"Warning: Google Search grounding requires a Gemini model, but the fast "
                f"model is '{self.fast_model_name}'. Falling back to scholarly database search."
            )
            return "s2"
        return backend

    def _safe_print(self, message: str):
        """Thread-safe printing to prevent garbled output."""
        with self.print_lock:
            print(message)

    def run(
        self,
        outline_path: str,
        cutoff_date: Optional[str] = None,
    ):
        if not cutoff_date:
            cutoff_date = datetime.datetime.now().strftime("%Y-%m")

        self._safe_print(
            f"🚀 Starting Fast Hybrid Literature Agent (Cutoff: {cutoff_date}, Threads: {self.max_workers})..."
        )

        with open(outline_path, "r") as f:
            outline = json.load(f)

        outline_output_path = f"{self.output_dir}/outline_v1.json"
        unique_papers_registry: Dict[str, PaperData] = {}
        processed_tasks = set()

        # --- PHASE 1: DISCOVERY & RETRIEVAL ---
        self._safe_print(f"📚 Phase 1: Analyzing Plan & Executing Search Tasks...")

        search_tasks = self._collect_search_tasks(outline)
        self._safe_print(
            f"    -> Identified {len(search_tasks)} distinct search tasks."
        )

        self._execute_tasks_in_parallel(
            search_tasks, outline, unique_papers_registry, processed_tasks, cutoff_date
        )

        self._safe_print(
            f"\n✨ Final Count: {len(unique_papers_registry)} unique enriched papers retrieved "
            f"({len(self.verification_log['recovered'])} recovered, "
            f"{len(self.verification_log['rejected'])} candidates rejected)."
        )
        self._save_verification_log(len(unique_papers_registry))

        # --- PHASE 2: ASSET GENERATION ---
        final_papers_list = list(unique_papers_registry.values())
        bibtex_str = self._generate_bibtex(final_papers_list)
        citation_map = self._generate_citation_map(final_papers_list)

        self._safe_print("🔄 Injecting citations back into Outline...")
        updated_outline = self._inject_citations_into_outline(
            outline, unique_papers_registry
        )

        # --- PHASE 3: WRITING ---
        self._safe_print("✍️  Phase 3: Writing Introduction & Related Work...")
        latex_content = self._synthesize_content(
            updated_outline, final_papers_list, cutoff_date
        )

        # --- PHASE 4: SAVING ---
        self._save_outputs(
            latex_content,
            bibtex_str,
            citation_map,
            updated_outline,
            outline_output_path,
        )
        self._safe_print(
            f"✅ Done! Literature Review Outputs saved to {self.output_dir}/"
        )

    def _execute_tasks_in_parallel(
        self, tasks, outline, registry, processed_tasks_set, cutoff_date
    ):
        """Run a batch of search tasks concurrently."""
        active_tasks = []
        for task in tasks:
            task_id = f"{task['section']}:{task['focus']}"
            if task_id in processed_tasks_set:
                continue
            active_tasks.append(task)
            processed_tasks_set.add(task_id)

        if not active_tasks:
            return

        backend_label = (
            "Google Search"
            if self.search_backend == "google"
            else "scholarly databases"
        )
        self._safe_print(
            f"   🔎 Candidate discovery via {backend_label} (10 threads), verifying "
            f"candidates in the background ({self.max_workers} workers: "
            f"Semantic Scholar / OpenAlex / arXiv)..."
        )

        # Candidates are verified as soon as their search task returns, while
        # other tasks are still discovering. Rate limiting and source fallback
        # are handled inside utils.scholar_utils.
        verify_futures = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="verify"
        ) as verifier:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_task = {
                    executor.submit(
                        self._process_search_task, task, outline, cutoff_date
                    ): task
                    for task in active_tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        cands = future.result() or []
                    except Exception as exc:
                        self._safe_print(
                            f"      ❌ Candidate discovery failed for '{task['focus'][:20]}...': {exc}"
                        )
                        continue
                    for cand_item in cands:
                        verify_futures.append(
                            verifier.submit(
                                self._enrich_and_register_paper,
                                cand_item,
                                task,
                                registry,
                                cutoff_date,
                            )
                        )
            self._safe_print(
                f"   📊 Discovery finished; waiting for {sum(not f.done() for f in verify_futures)} "
                f"of {len(verify_futures)} candidate verifications..."
            )
            for future in as_completed(verify_futures):
                try:
                    future.result()
                except Exception as exc:  # defensive: workers handle their own errors
                    self._safe_print(f"      ❌ Verification worker error: {exc}")

    def _collect_search_tasks(self, outline: Dict) -> List[Dict]:
        tasks = []
        plan = outline.get("intro_related_work_plan", {})

        # 1. Introduction Macro-Tasks
        intro_strat = plan.get("introduction_strategy", {})
        for direction in intro_strat.get("search_directions", []):
            tasks.append(
                {
                    "section": "Introduction",
                    "focus": direction,
                    "context": f"Hook: {intro_strat.get('hook_hypothesis', '')}. Gap: {intro_strat.get('problem_gap_hypothesis', '')}",
                    "search_type": "exploration",
                }
            )

        # 2. Related Work Micro-Tasks
        rw_strat = plan.get("related_work_strategy", {})
        for sub in rw_strat.get("subsections", []):
            queries = sub.get("limitation_search_queries", [])
            if not queries:
                queries = [
                    f"{sub.get('subsection_title', '')} {sub.get('methodology_cluster', '')}"
                ]

            for q in queries:
                tasks.append(
                    {
                        "section": f"Related Work: {sub.get('subsection_title', 'General')}",
                        "focus": q,
                        "context": f"Mission: {sub.get('sota_investigation_mission', '')}. Hypothesis: {sub.get('limitation_hypothesis', '')}",
                        "search_type": "exploration",
                    }
                )

        # 3. Targeted Tasks (Must-Haves from Section Plan)
        blueprint = outline.get("section_plan", [])
        for section in blueprint:
            section_title = section.get("section_title", "Unknown Section")
            for subsection in section.get("subsections", []):
                sub_title = subsection.get("subsection_title", "General")
                bullets = subsection.get("content_bullets", [])
                context_str = " ".join(bullets)

                hints = subsection.get("citation_hints", [])
                for hint in hints:
                    tasks.append(
                        {
                            "section": f"{section_title} - {sub_title}",
                            "focus": hint,
                            "context": f"Must-have citation for section covering: {context_str}",
                            "search_type": "targeted",
                        }
                    )

        return tasks

    def _inject_citations_into_outline(
        self, outline: Dict, registry: Dict[str, PaperData]
    ) -> Dict:
        # Group keys by section name
        section_to_keys = {}
        for p in registry.values():
            sec = p.found_in_section
            if sec not in section_to_keys:
                section_to_keys[sec] = []
            section_to_keys[sec].append(p.citation_key)

        plan = outline.get("intro_related_work_plan", {})

        # Introduction
        intro_strat = plan.get("introduction_strategy", {})
        intro_keys = section_to_keys.get("Introduction", [])
        intro_strat["citation_candidates"] = list(set(intro_keys))

        # Related Work
        rw_strat = plan.get("related_work_strategy", {})
        for sub in rw_strat.get("subsections", []):
            key = f"Related Work: {sub.get('subsection_title', 'General')}"
            sub["citation_candidates"] = section_to_keys.get(key, [])

        # Other Sections
        blueprint = outline.get("section_plan", [])
        for section in blueprint:
            section_title = section.get("section_title", "Unknown Section")
            for sub in section.get("subsections", []):
                sub_title = sub.get("subsection_title", "General")
                key = f"{section_title} - {sub_title}"
                sub["citation_candidates"] = section_to_keys.get(key, [])
                if "citation_hints" in sub:
                    del sub["citation_hints"]

        return outline

    def _process_search_task(self, task, outline, cutoff_date) -> list:
        if task.get("search_type") == "targeted":
            self._safe_print(
                f"   🎯 Targeted Search [{task['section']}]: '{task['focus']}'"
            )
        else:
            self._safe_print(
                f"   🔎 Exploration [{task['section']}]: '{task['focus']}'"
            )

        retry_count = 0
        while retry_count < 3:
            try:
                candidates = self._discover_candidates(task, outline, cutoff_date)
                if candidates:
                    return candidates
                retry_count += 1
            except Exception as e:
                retry_count += 1
        return []

    @staticmethod
    def _cand_field(cand, name, default=None):
        if isinstance(cand, dict):
            return cand.get(name, default)
        return getattr(cand, name, default)

    def _reject(self, title, year, task, reason, detail=""):
        entry = {
            "title": title,
            "year": year,
            "section": task.get("section"),
            "reason": reason,
        }
        if detail:
            entry["detail"] = detail
        with self.registry_lock:
            self.verification_log["rejected"].append(entry)
        self._safe_print(f"      -> 🚫 Rejected ({reason}): {title}")

    def _save_verification_log(self, verified_count: int):
        path = os.path.join(self.output_dir, "candidate_verification.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "verified_papers": verified_count,
                    "recovery_attempts": self._recoveries_used,
                    **self.verification_log,
                },
                f,
                indent=4,
            )

    def _enrich_and_register_paper(self, cand, task, registry, cutoff_date):
        cand_title = self._cand_field(cand, "title")
        cand_year = self._cand_field(cand, "year")
        cand_reason = self._cand_field(cand, "reason", "") or ""
        if not cand_title:
            return

        norm_title = self._normalize_title(cand_title)

        # Skip titles already registered or being verified by another worker.
        with self.registry_lock:
            if norm_title in registry or norm_title in self._inflight_titles:
                return
            self._inflight_titles.add(norm_title)

        try:
            s2_data = title_search(
                cand_title, cand_year, cutoff_date, require_abstract=True
            )
            recovered = False
            if s2_data is None:
                s2_data, why = self._recover_candidate(
                    cand_title, cand_year, cand_reason, task, cutoff_date
                )
                if s2_data is None:
                    self._reject(cand_title, cand_year, task, why)
                    return
                recovered = True

            if s2_data:
                if not self._is_paper_allowed(s2_data, cutoff_date):
                    self._reject(cand_title, cand_year, task, "after_cutoff")
                    return

                if not s2_data.get("abstract", ""):
                    self._reject(cand_title, cand_year, task, "no_abstract")
                    return

                authors_list = [a["name"] for a in s2_data.get("authors", [])]
                j_info = s2_data.get("journal") or {}

                abstract_text = s2_data.get("abstract") or ""
                if len(abstract_text) > 1500:
                    abstract_text = abstract_text[:1500] + "... [Truncated]"

                full_obj = PaperData(
                    citation_key=self._generate_key(
                        s2_data.get("authors", []),
                        s2_data.get("year"),
                        s2_data["title"],
                    ),
                    title=s2_data["title"],
                    authors=authors_list,
                    venue=s2_data.get("venue") or "arXiv",
                    year=s2_data.get("year") or cand_year,
                    abstract=abstract_text,
                    citation_count=s2_data.get("citationCount") or 0,
                    found_in_section=task["section"],
                    reason=cand_reason,
                    journal=j_info.get("name"),
                    volume=j_info.get("volume"),
                    pages=j_info.get("pages"),
                    publication_date=s2_data.get("publicationDate"),
                    arxiv_id=s2_data.get("arxivId"),
                    doi=s2_data.get("doi"),
                )

                final_norm_title = self._normalize_title(full_obj.title)

                with self.registry_lock:
                    if final_norm_title not in registry:
                        registry[final_norm_title] = full_obj
                        if recovered:
                            self.verification_log["recovered"].append(
                                {
                                    "candidate_title": cand_title,
                                    "matched_title": full_obj.title,
                                    "venue": full_obj.venue,
                                    "year": full_obj.year,
                                    "section": task.get("section"),
                                }
                            )
                        self._safe_print(
                            f"      -> 📚 Added enriched paper"
                            f"{' (recovered)' if recovered else ''}: {full_obj.title}"
                            f" [{s2_data.get('source', '?')}]"
                        )
        except Exception as e:
            self._reject(cand_title, cand_year, task, "error", str(e))

    def _recover_candidate(self, cand_title, cand_year, cand_reason, task, cutoff_date):
        """LLM-assisted recovery for a candidate whose title was not found.

        Fixed loop (works without tool calling, so small open models can run
        it): the LLM proposes corrected titles / queries, the databases are
        searched, and the LLM picks the matching real record or declares the
        candidate non-existent. Returns (record or None, rejection reason).
        """
        with self.registry_lock:
            if self._recoveries_used >= self.max_citation_recoveries:
                return None, "not_found"
            self._recoveries_used += 1

        try:
            query_prompt = f"""
            A citation candidate could not be found in scholarly databases (Semantic Scholar,
            OpenAlex, arXiv) by its title.

            CANDIDATE TITLE: {cand_title}
            CANDIDATE YEAR: {cand_year}
            WHY IT WAS SUGGESTED: {cand_reason}
            SEARCH TASK: {task.get('focus', '')}
            CONTEXT: {task.get('context', '')}

            The title may be slightly wrong, abbreviated, the title of a different version
            (preprint vs. published), or the paper may not exist at all.
            Propose up to 3 search queries to find the real paper: first your best guess of
            its EXACT correct title (if you know it), then short keyword queries.
            If you are confident no such paper exists, return an empty list.

            Return ONLY json in a ```json``` block: {{"queries": ["...", "..."]}}
            """
            response = call_llm_with_contents(
                model_name=self.fast_model_name,
                contents=[query_prompt],
                generation_configs={"temperature": 0.2},
            )
            parsed = response["parsed_response"]
            queries = parsed.get("queries", []) if isinstance(parsed, dict) else parsed
            queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:3]
            if not queries:
                return None, "nonexistent"

            found: Dict[str, Dict] = {}
            for i, q in enumerate(queries):
                hit = title_search(q, cand_year, cutoff_date, require_abstract=True)
                if hit:
                    found.setdefault(self._normalize_title(hit["title"]), hit)
                if i < 2:  # keyword searches may use the OpenAlex daily budget
                    for r in keyword_search(q, limit=5, cutoff_date=cutoff_date):
                        if r.get("abstract"):
                            found.setdefault(self._normalize_title(r["title"]), r)
            if not found:
                return None, "not_found_after_recovery"

            records = list(found.values())[:12]
            listing = []
            for i, r in enumerate(records):
                abstract = (r.get("abstract") or "").replace("\n", " ")[:300]
                listing.append(
                    f"[{i}] {r['title']} ({r.get('year') or 'n.d.'}; "
                    f"{r.get('venue') or 'unknown venue'}; "
                    f"{r.get('citationCount') or 0} citations)\n    {abstract}"
                )
            select_prompt = f"""
            We are looking for the real paper behind this citation candidate:
            CANDIDATE TITLE: {cand_title}
            CANDIDATE YEAR: {cand_year}
            WHY IT WAS SUGGESTED: {cand_reason}

            SEARCH RESULTS:
            {chr(10).join(listing)}

            Which search result IS this paper (the same work, possibly under a corrected or
            published title)? Do NOT pick a merely related paper. If it is a preprint and the
            published version is also listed, pick the published version.
            If none of the results is the same work, use null.

            Return ONLY json in a ```json``` block:
            {{"match_index": <integer or null>, "explanation": "<one sentence>"}}
            """
            response = call_llm_with_contents(
                model_name=self.fast_model_name,
                contents=[select_prompt],
                generation_configs={"temperature": 0.0},
            )
            parsed = response["parsed_response"]
            index = parsed.get("match_index") if isinstance(parsed, dict) else None
            if isinstance(index, str) and index.strip().isdigit():
                index = int(index)
            if isinstance(index, int) and 0 <= index < len(records):
                record = records[index]
                self._safe_print(
                    f"      -> 🩹 Recovered '{cand_title}' as '{record['title']}'"
                )
                return record, ""
            return None, "nonexistent"
        except Exception as e:
            self._safe_print(f"      ⚠️ Recovery failed for '{cand_title}': {e}")
            return None, "not_found"

    def _is_paper_allowed(self, s2_data: Dict, cutoff_date_str: str) -> bool:
        if not cutoff_date_str:
            return True

        c_parts = [int(x) for x in cutoff_date_str.split("-")]
        c_y = c_parts[0]
        c_m = c_parts[1] if len(c_parts) > 1 else 12
        c_d = c_parts[2] if len(c_parts) > 2 else 1

        p_y, p_m, p_d = None, None, None
        if s2_data.get("publicationDate"):
            try:
                p_parts = [int(x) for x in str(s2_data["publicationDate"]).split("-")]
                p_y = p_parts[0]
                p_m = p_parts[1] if len(p_parts) > 1 else None
                p_d = p_parts[2] if len(p_parts) > 2 else None
            except:
                pass

        if p_y is None and s2_data.get("year"):
            p_y = int(s2_data["year"])

        if p_y is None:
            return True
        if p_y > c_y:
            return False
        if p_y < c_y:
            return True
        if p_m is None:
            return True
        if p_m > c_m:
            return False
        if p_m < c_m:
            return True
        if p_d is None:
            return True
        if p_d > c_d:
            return False

        return True

    def _discover_candidates(self, task, outline, cutoff_date) -> List[CandidatePaper]:
        if self.search_backend == "google":
            return self._discover_candidates_google(task, outline, cutoff_date)
        return self._discover_candidates_s2(task, outline, cutoff_date)

    def _discover_candidates_google(self, task, outline, cutoff_date) -> list:
        intro_plan = outline.get("intro_related_work_plan", {}).get(
            "introduction_strategy", {}
        )
        core_problem = intro_plan.get("problem_gap_hypothesis", "N/A")

        if task.get("search_type") == "targeted":
            prompt = f"""
            Identify the SPECIFIC academic paper described by this hint: "{task['focus']}".
            
            CONTEXT: {task['context']}
            CUTOFF DATE: Published BEFORE {cutoff_date}
            
            INSTRUCTIONS:
            1. Use Google Search to find the EXACT title and year of this paper.
            2. If the hint refers to a dataset, model architecture, or baseline (e.g., "research paper introducing..."), you MUST find the primary, original paper that introduced it.
            3. Return EXACTLY 1 candidate paper that matches best.

            YOU MUST return the results in json format (```json content```) following this schema:
            {DiscoveryResult.model_json_schema()}
            """
        else:
            prompt = f"""
            Find 10-15 highly relevant, influential academic papers matching the search task. Do not force results; only return strictly relevant papers.
            
            SEARCH TASK: {task['focus']}
            CONTEXT FOR TASK: {task['context']}
            BROADER PROJECT PROBLEM: {core_problem}
            CUTOFF DATE: Published BEFORE {cutoff_date}
            
            INSTRUCTIONS:
            1. Use Google Search to find REAL papers (ArXiv, CVPR, NeurIPS, ICML, ICLR, etc.).
            2. Only return papers published in top-tier conferences or journals, or highly-cited works, or the exact state-of-the-art relevant to the task.

            YOU MUST return the results in json format (```json content```) following this schema:
            {DiscoveryResult.model_json_schema()}
            """

        response_dict = call_llm_with_contents(
            model_name=self.fast_model_name,
            contents=[prompt],
            generation_configs={
                "tools": web_search_tools(self.fast_model_name),
                "temperature": 0.1 if task.get("search_type") == "targeted" else 0.4,
            },
        )

        if response_dict["parsed_response"]:
            return response_dict["parsed_response"].get("candidates", [])
        return []

    def _project_brief(self, outline) -> str:
        """Short description of the project used to anchor queries and relevance."""
        intro_plan = outline.get("intro_related_work_plan", {}).get(
            "introduction_strategy", {}
        )
        parts = [
            intro_plan.get("hook_hypothesis", ""),
            intro_plan.get("problem_gap_hypothesis", ""),
        ]
        idea = (self.context_files.get("idea.md") or "").strip()
        if idea:
            parts.append(idea[:1200] + ("..." if len(idea) > 1200 else ""))
        return "\n".join(p for p in parts if p)

    def _generate_queries(self, task, project_brief: str) -> tuple[List[str], List[str]]:
        """Asks the LLM for short keyword queries plus exact titles of
        well-known papers. Returns (queries, known_titles)."""
        targeted = task.get("search_type") == "targeted"
        if targeted:
            # Targeted hints name a specific method/dataset/tool: queries must
            # be about that thing only, not the project (which drags the
            # search toward papers that merely use it).
            prompt = f"""
            Find the ORIGINAL paper that introduced the thing named in this hint:
            "{task['focus']}"

            Background (only to understand acronyms and which thing is meant):
            {project_brief[:600]}

            - "known_titles": the exact title of that original paper, if you know it. The
              title often differs from the method's name. Else [].
            - "queries": 2-3 keyword queries (2-6 words) about that thing ONLY:
              its name; its acronym written out in full words; and what it is.
              If the hint names a variant or component of a model (e.g. "ViT-L version of
              CLIP"), query the base model itself ("CLIP contrastive language image").
              Do not add words about the project or any other topic.

            EXAMPLE: hint "research paper introducing 'BatchNorm layers'"
            {{"known_titles": ["Batch Normalization: Accelerating Deep Network Training by
              Reducing Internal Covariate Shift"],
              "queries": ["batch normalization internal covariate shift"]}}

            Return ONLY json in a ```json``` block: {{"queries": [...], "known_titles": [...]}}
            """
        else:
            prompt = f"""
            You write search queries for an academic search engine (keyword matching on
            titles and abstracts), to find papers for the project below.

            PROJECT:
            {project_brief}

            SEARCH TASK: {task['focus']}
            CONTEXT FOR TASK: {task['context']}
            - "known_titles": exact titles of up to 5 well-known papers on this exact topic
              that you are certain exist (else []).
            - "queries": 3-5 keyword queries.

            QUERY RULES:
            - 2-6 words each: technical noun phrases, the way paper titles are written.
            - Every query must name the specific research area of the project (e.g. the task,
              modality or model family), so results cannot drift to unrelated fields.
            - No instruction words: never use "find", "identify", "papers", "research",
              "highly cited", "foundational", "impact", "survey of", "limitations of".
            - Vary the wording: synonyms, related task names, method names.

            EXAMPLE (for a different project):
            Task: "Find highly cited papers establishing the real-world impact of
            hallucination in LLM summarization"
            {{"queries": ["abstractive summarization hallucination",
                          "factual consistency summarization evaluation",
                          "faithfulness large language model summaries"],
              "known_titles": ["On Faithfulness and Factuality in Abstractive Summarization"]}}

            Return ONLY json in a ```json``` block: {{"queries": [...], "known_titles": [...]}}
            """
        response = call_llm_with_contents(
            model_name=self.fast_model_name,
            contents=[prompt],
            generation_configs={"temperature": 0.2},
        )
        parsed = response["parsed_response"]
        if not isinstance(parsed, dict):
            parsed = {"queries": parsed if isinstance(parsed, list) else []}
        queries = [clean_search_query(q) for q in parsed.get("queries") or [] if isinstance(q, str)]
        known = [t.strip() for t in parsed.get("known_titles") or [] if isinstance(t, str) and t.strip()]

        if targeted:
            # Names quoted in the hint (e.g. 'Segment Anything Model (SAM)') are
            # the most reliable query of all.
            for name in re.findall(r"['\"‘“]([^'\"’”]{3,120})['\"’”]", task["focus"]):
                queries.insert(0, clean_search_query(name))
        queries = [q for q in dict.fromkeys(queries) if q]
        if not queries:
            fallback = clean_search_query(task["focus"])
            queries = [fallback] if fallback else []
        return queries[: 3 if targeted else 5], known[: 1 if targeted else 5]

    def _discover_candidates_s2(self, task, outline, cutoff_date) -> list:
        """Model-agnostic discovery: the LLM writes short keyword queries (and
        names well-known papers), scholarly databases are searched, and the LLM
        scores each real result for relevance to the project."""
        targeted = task.get("search_type") == "targeted"
        project_brief = self._project_brief(outline)
        queries, known_titles = self._generate_queries(task, project_brief)
        self._safe_print(
            f"      queries for '{task['focus'][:50]}': {queries}"
            + (f" + known titles {known_titles}" if known_titles else "")
        )

        # Search: verified known titles first, then keyword results.
        found: Dict[str, Dict] = {}
        for title in known_titles:
            hit = title_search(title, None, cutoff_date, require_abstract=True)
            if hit:
                found.setdefault(self._normalize_title(hit["title"]), hit)
            elif targeted:
                # Near-miss titles still make good keyword queries.
                guess = clean_search_query(title)
                if guess and guess not in queries:
                    queries.insert(0, guess)
        if targeted:
            # The original paper often has a short title ("Segment Anything")
            # that keyword search ranks below its many derivatives; a title
            # lookup (free, most-cited first) finds it directly.
            for q in title_lookup_variants(queries[:3])[:4]:
                for r in title_candidates(q, limit=4, cutoff_date=cutoff_date):
                    if r.get("abstract"):
                        found.setdefault(self._normalize_title(r["title"]), r)
        per_query = 8 if targeted else 10
        for q in queries:
            for r in keyword_search(q, limit=per_query, cutoff_date=cutoff_date):
                if r.get("abstract"):
                    found.setdefault(self._normalize_title(r["title"]), r)
        if not found:
            return []
        records = list(found.values())[:30]

        listing = []
        for i, r in enumerate(records):
            abstract = (r.get("abstract") or "").replace("\n", " ")
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            listing.append(
                f"[{i}] {r['title']} ({r.get('year') or 'n.d.'}; "
                f"{r.get('venue') or 'unknown venue'}; "
                f"{r.get('citationCount') or 0} citations)\n    {abstract}"
            )

        if targeted:
            rubric = (
                f'Score 3 ONLY for the original paper that INTRODUCED what is named in '
                f'"{task["focus"]}". A paper that merely uses, applies, evaluates or mentions '
                "it scores 0, even if the name appears in its title. If the hint names a "
                "variant, size, version or component of a model (e.g. 'ViT-L backbone of "
                "CLIP'), the original paper of the base model scores 3. If none of the "
                "results is the original paper, score them all 0."
            )
        else:
            rubric = (
                "3 = directly about the project's research area and this search task; "
                "2 = same research area, useful context for the task; "
                "1 = tangential (shares a technique but a different field or task); "
                "0 = unrelated. Be strict: papers from other fields (medicine, "
                "recommendation, finance, ...) are 0 even if they share words."
            )
        select_prompt = f"""
            PROJECT:
            {project_brief}

            SEARCH TASK: {task['focus']}
            CONTEXT FOR TASK: {task['context']}

            SEARCH RESULTS:
            {chr(10).join(listing)}

            Rate how relevant each search result is for citing in this project's paper.
            {rubric}

            Return ONLY json in a ```json``` block, one entry per result index:
            {{"scores": [{{"index": 0, "score": 3, "reason": "<few words>"}}, ...]}}
            """
        response = call_llm_with_contents(
            model_name=self.fast_model_name,
            contents=[select_prompt],
            generation_configs={"temperature": 0.0},
        )
        min_score = 3 if targeted else MIN_RELEVANCE_SCORE
        selected = []
        for item in parse_relevance_scores(response["parsed_response"]):
            index, score = item["index"], item["score"]
            if 0 <= index < len(records) and score >= min_score:
                selected.append((score, records[index], item.get("reason", "")))
        selected.sort(key=lambda x: (x[0], x[1].get("citationCount") or 0), reverse=True)
        limit = 1 if targeted else 15
        dropped = len(records) - len(selected)
        if dropped:
            self._safe_print(
                f"      kept {min(len(selected), limit)}/{len(records)} results for "
                f"'{task['focus'][:50]}' (relevance >= {min_score})"
            )
        return [
            {"title": r["title"], "year": r.get("year"), "reason": reason or task["focus"]}
            for _, r, reason in selected[:limit]
        ]

    def _synthesize_content(self, outline, papers: List[PaperData], cutoff_date: str):
        paper_keys = [p.citation_key for p in papers]
        papers_payload = [p.model_dump() for p in papers]

        context_payload = {
            "template.tex": self.context_files.get("template.tex", ""),
            "intro_related_work_plan": outline.get("intro_related_work_plan", {}),
            "project_idea": self.context_files.get("idea.md", ""),
            "project_experimental_log": self.context_files.get(
                "experimental_log.md", ""
            ),
            "citation_checklist": paper_keys,
            "collected_papers": papers_payload,
        }

        writer_system_prompt = literature_review_agent_writter_prompt.format(
            paper_count=len(papers),
            min_cite_paper_count=int(len(papers) * 0.9),
            cutoff_date=cutoff_date,
        )

        user_prompt = "Generate the LaTeX for Introduction and Related Work sections."
        response_dict = call_llm_with_contents(
            model_name=self.model_name,
            contents=[user_prompt, json.dumps(context_payload)],
            generation_configs={"system_instruction": writer_system_prompt},
            result_parsing_func=parse_latex,
        )
        return response_dict["parsed_response"]

    def _generate_key(self, authors_raw, year, title):
        if not authors_raw:
            first_author = "Unknown"
        elif isinstance(authors_raw[0], dict):
            first_author = authors_raw[0]["name"].split()[-1]
        else:
            first_author = str(authors_raw[0]).split()[-1]

        clean_author = re.sub(r"[^a-zA-Z]", "", first_author).capitalize()
        year_str = str(year) if year else "2024"
        clean_title_str = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
        words = clean_title_str.split()
        stopwords = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "for",
            "to",
            "of",
            "and",
            "is",
            "are",
            "with",
            "by",
            "study",
        }
        meaningful = [w.capitalize() for w in words if w not in stopwords]
        title_part = "".join(meaningful[:2]) if meaningful else "Paper"
        return f"{clean_author}{year_str}{title_part}"

    def _generate_bibtex(self, papers: List[PaperData]) -> str:
        entries = []
        seen_keys = set()
        for p in papers:
            base_key = p.citation_key
            unique_key = base_key
            suffix = "a"
            while unique_key in seen_keys:
                unique_key = f"{base_key}{suffix}"
                suffix = chr(ord(suffix) + 1)
            p.citation_key = unique_key
            seen_keys.add(unique_key)

            author_str = " and ".join(p.authors) if p.authors else "Unknown"
            is_preprint = not p.journal and (p.venue or "").lower().startswith("arxiv")
            entry_type = "article" if (p.journal or is_preprint) else "inproceedings"

            entry = f"@{entry_type}{{{p.citation_key},\n  title={{{p.title}}},\n  author={{{author_str}}},\n"
            if p.journal:
                entry += f"  journal={{{p.journal}}},\n"
            elif is_preprint:
                preprint = f"arXiv preprint arXiv:{p.arxiv_id}" if p.arxiv_id else "arXiv preprint"
                entry += f"  journal={{{preprint}}},\n"
            else:
                entry += f"  booktitle={{{p.venue}}},\n"
            entry += f"  year={{{p.year}}}"
            if p.volume:
                entry += f",\n  volume={{{p.volume}}}"
            if p.pages:
                entry += f",\n  pages={{{p.pages}}}"
            entry += "\n}"
            entries.append(entry)
        return "\n\n".join(entries)

    def _generate_citation_map(self, papers: List[PaperData]) -> Dict[str, Dict]:
        return {
            p.citation_key: {
                "citation_key": p.citation_key,
                "title": p.title,
                "authors": p.authors,
                "venue": p.venue,
                "year": p.year,
                "abstract": p.abstract.strip(),
            }
            for p in papers
            if p.abstract
        }

    def _save_outputs(self, latex, bibtex, citation_map, outline, outline_path):
        with open(
            f"{self.output_dir}/updated_template.tex", "w", encoding="utf-8"
        ) as f:
            f.write(latex)
        with open(f"{self.output_dir}/references.bib", "w", encoding="utf-8") as f:
            f.write(bibtex)
        with open(f"{self.output_dir}/citation_map.json", "w", encoding="utf-8") as f:
            json.dump(citation_map, f, indent=4)
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=4)

    def _load_context_files(self) -> Dict[str, str]:
        loaded = {}
        for name, path in self.context_paths.items():
            if os.path.exists(path):
                with open(path, "r") as f:
                    loaded[name] = f.read()
            else:
                loaded[name] = ""
        return loaded

    def _normalize_title(self, title):
        return re.sub(r"[^a-z0-9]", "", title.lower())
