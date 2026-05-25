#!/usr/bin/env python3
"""Render the seven data-construction prompts into a LaTeX appendix fragment.

The deployed prompts (datagen/**/prompts/*.md) use Unicode math glyphs for
readability; this script transcribes them to ASCII so the fragment compiles
under pdflatex + inputenc, then wraps each prompt in a tcolorbox+lstlisting
block matching the LaTeX prompt-listing style.

Usage:
    python scripts/render_prompts_to_latex.py > /path/to/fragment.tex

The fragment is a single \\subsection ready to \\input or paste into the
\\section{Prompts}. Re-run after editing any prompt to keep the
appendix aligned with the repository (modulo the ASCII transcription
this script performs).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "datagen"

PROMPTS = [
    ("rules/prompts/gen.md",                      "Rule Generation (\\texttt{rules/gen})"),
    ("rules/prompts/qc.md",                        "Rule Quality Control (\\texttt{rules/qc})"),
    ("test_sessions/prompts/gen.md",               "Test-Session Generation (\\texttt{test\\_sessions/gen})"),
    ("test_sessions/prompts/qc.md",                "Test-Session QC: Agent Trace Simulation (\\texttt{test\\_sessions/qc})"),
    ("test_sessions/prompts/refine.md",            "Test-Session Repair (\\texttt{test\\_sessions/refine})"),
    ("learning_sessions/skeleton/prompts/gen.md",  "Learning-Session Skeleton (\\texttt{learning\\_sessions/skeleton})"),
    ("learning_sessions/fill/prompts/gen.md",      "Learning-Session Fill (\\texttt{learning\\_sessions/fill})"),
]

# Unicode -> ASCII transcription (LaTeX typesetting only; repo prompts keep Unicode).
SYM = {
    "—": "--", "–": "-", "→": "->", "≥": ">=", "≤": "<=",
    "∈": " in ", "×": "x", "≈": "~=", "⊆": "subseteq", "·": "/", "§": "Sec.",
}


def asciify(t: str) -> str:
    for u, r in SYM.items():
        t = t.replace(u, r)
    return t


def main() -> None:
    out = [
        r"\subsection{Data Construction Prompts}",
        r"\label{app:data_construction_prompts}",
        "",
        r"The seven prompts below drive the construction pipeline (Appendix~\ref{sec:appendix_construction}): standing-rule generation and two-axis QC, rule-bound test-session generation, QC trace simulation, and repair, and learning-session skeleton and fill. Double-brace placeholders (e.g.\ \texttt{\{\{RULE\_JSON\}\}}) are filled at call time; few-shot pools and persona or tool context are elided where marked. Non-ASCII glyphs in the deployed prompts are transcribed to ASCII here for typesetting.",
        "",
    ]
    for rel, title in PROMPTS:
        text = asciify((BASE / rel).read_text(encoding="utf-8").rstrip("\n"))
        out += [r"\begin{tcolorbox}[title={%s}]" % title, r"\begin{lstlisting}",
                text, r"\end{lstlisting}", r"\end{tcolorbox}", ""]
    frag = "\n".join(out)
    non_ascii = [c for c in frag if ord(c) > 127]
    if non_ascii:
        raise SystemExit(f"non-ASCII remains: {set(non_ascii)}")
    print(frag, end="")


if __name__ == "__main__":
    main()
