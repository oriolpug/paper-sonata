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


def test_ensure_figures_included_inserts_before_conclusion():
    from utils.latex_cleanup import ensure_figures_included

    tex = "\\section{Method}\nText.\n\\section{Conclusion}\nEnd.\n\\end{document}"
    figs = [{"name": "overview.png", "caption": "Our model & 50% faster"},
            {"name": "plot.pdf", "caption": "Already used"}]
    tex_with_plot = tex.replace("Text.", "Text. \\includegraphics[width=\\linewidth]{figures/plot.pdf}")
    out, changes = ensure_figures_included(tex_with_plot, figs)
    assert changes == ["inserted missing figure 'overview.png'"]
    assert out.index("figures/overview.png") < out.index("\\section{Conclusion}")
    assert "Our model \\& 50\\% faster" in out and "\\label{fig:overview}" in out
    assert out.count("plot.pdf") == 1
    assert ensure_figures_included(out, figs) == (out, [])
    assert ensure_figures_included(tex, []) == (tex, [])


def test_example_phrases_and_copied_examples():
    from utils.latex_cleanup import example_phrases, find_copied_examples

    for conf in ("cvpr2025", "iclr2025"):
        guide = open(f"templates/{conf}/guidelines.md").read()
        tmpl = open(f"templates/{conf}/template.tex").read()
        cells, placeholders = example_phrases(guide, tmpl)
        assert cells and "Methodology here." in placeholders
        assert "Method" not in cells
    guide = open("templates/cvpr2025/guidelines.md").read()
    tmpl = open("templates/cvpr2025/template.tex").read()
    assert {"Frobnability", "Frumpy", "Frobbly"} <= set(example_phrases(guide)[0])
    bad = "Method & Frobnability \\\\ AVSBench & Frumpy"
    assert "copied example" in find_copied_examples(bad, guide)[0]
    assert find_copied_examples("Method & J \\\\ TSAM & 43.43", guide, tmpl) == []
    assert "placeholders" in find_copied_examples("Conclusion here.", guide, tmpl)[0]


def test_document_parser_rejects_invalid_documents():
    from utils.llm_parsing import make_document_parser

    parse = make_document_parser("", validate=lambda d: ["bad"] if "Frob" in d else [])
    assert parse("```latex\n\\begin{document}Frob\\end{document}\n```") is None
    assert parse("```latex\n\\begin{document}ok\\end{document}\n```") == "\\begin{document}ok\\end{document}"


def test_fix_markdown_artifacts():
    from utils.latex_cleanup import fix_markdown_artifacts as f

    md = ("The branch works as follows:\n"
          "1. **Sequential Processing:** reads frames.\n"
          "2. **Cached Memory & Adapters:** stores state.\n"
          "Then more text.")
    out = f(md)
    assert "\\begin{enumerate}\n  \\item \\textbf{Sequential Processing:} reads frames." in out
    assert "\\textbf{Cached Memory \\& Adapters:}" in out and "\\end{enumerate}\nThen more text." in out

    table = "\\begin{tabular}{lc}\nSetting & Mix(S+U) J(%) \\\\\nA & 49.01 \\\\\n\\end{tabular}"
    out = f(table)
    assert "Setting & Mix(S+U) J(\\%)" in out and "A & 49.01" in out

    eq = "\\begin{align} a &= b \\\\ c &= d \\end{align} and R&D gains 5% % a comment"
    out = f(eq)
    assert "a &= b" in out and "R\\&D" in out and "5\\%" in out and " % a comment" in out
    assert f("%%%%%%%%% INTRODUCTION %%%%%%%%%") == "%%%%%%%%% INTRODUCTION %%%%%%%%%"
    assert f("Already \\& and 3\\% fine") == "Already \\& and 3\\% fine"
    assert f("$2 * 3$ and a*b") == "$2 * 3$ and a*b"


def test_markdown_lists_not_converted_inside_tables():
    from utils.latex_cleanup import fix_markdown_artifacts as f

    table = "\\begin{tabular}{lc}\n    - TB & 41.77 \\\\\n    - TMFL & 42.95 \\\\\n\\end{tabular}"
    assert f(table) == table
    assert "\\begin{itemize}" in f("Points:\n- one\n- two\n")


def test_resolve_figure_files():
    from utils.latex_cleanup import resolve_figure_files

    cap = "Overview of TSAM. Left: multimodal cues. Right: temporal branch."
    tex = ("Text.\n\\begin{figure}\n\\includegraphics[width=\\linewidth]{figures/tsam_overview.png}\n"
           f"\\caption{{{cap}}}\n\\label{{fig:a}}\n\\end{{figure}}\n"
           "\\begin{figure*}\n\\includegraphics{figures/invented_plot.png}\n\\caption{Accuracy plot}\n\\end{figure*}\n"
           "\\begin{figure}\n\\includegraphics{figures/real.png}\n\\caption{Real}\n\\end{figure}\n"
           "Inline \\includegraphics{figures/ghost.pdf} here.")
    out, changes = resolve_figure_files(
        tex, ["real.png", "cvpr_example_figure.png"],
        [{"name": "cvpr_example_figure.png", "caption": cap}, {"name": "real.png", "caption": "Real"}])
    assert "figures/cvpr_example_figure.png" in out and "tsam_overview" not in out
    assert "invented_plot" not in out and "Accuracy plot" not in out
    assert "figures/real.png" in out and "ghost.pdf" not in out
    assert len(changes) == 3
