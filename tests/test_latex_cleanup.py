from utils.latex_cleanup import clean_generated_latex, fix_citation_keys, strip_manual_section_numbers

KEYS = ["Li2020MstcnMultistage", "Kirillov2023SegmentAnything", "Perazzi2016BenchmarkDataset"]


def test_maps_near_miss_and_drops_unknown_keys():
    tex = r"See~\cite{Li2020Mst3nMultistage, Perazzi2016BenchmarkDataset, Totally2099Invented}."
    out, changes = fix_citation_keys(tex, KEYS)
    assert out == r"See~\cite{Li2020MstcnMultistage, Perazzi2016BenchmarkDataset}."
    assert any("Mst3n" in c for c in changes) and any("Invented" in c for c in changes)


def test_removes_cite_with_only_unknown_keys_and_keeps_options():
    out, _ = fix_citation_keys(r"Text~\cite{Nope2020Fake}. More \citep[p.~3]{Kirillov2023SegmentAnything}.", KEYS)
    assert out == r"Text. More \citep[p.~3]{Kirillov2023SegmentAnything}."


def test_deduplicates_after_mapping():
    out, _ = fix_citation_keys(r"\cite{Li2020Mst3nMultistage,Li2020MstcnMultistage}", KEYS)
    assert out == r"\cite{Li2020MstcnMultistage}"


def test_strip_manual_section_numbers():
    tex = "\\section{2. Related Work}\n\\subsection{2.1 Static Models}\n\\subsection*{IV. Results}\n\\section{Method 2.0}"
    assert strip_manual_section_numbers(tex) == (
        "\\section{Related Work}\n\\subsection{Static Models}\n\\subsection*{Results}\n\\section{Method 2.0}")


def test_clean_generated_latex_valid_input_unchanged():
    tex = r"\section{Intro} as in \cite{Kirillov2023SegmentAnything}."
    assert clean_generated_latex(tex, KEYS) == (tex, [])
