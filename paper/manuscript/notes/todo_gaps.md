# Pre-Submission Supplement Checklist

This file is the author-facing supplement checklist for turning the current draft into a submission-ready package.

## Priority 1: Required submission metadata

| Item | Status | Who should supply it | Required source material | Update targets |
| --- | --- | --- | --- | --- |
| Final author list and order | Resolved | Authors | Confirmed author names and order | `paper/manuscript/main.tex` |
| Affiliations | Resolved | Authors | Confirmed institution name | `paper/manuscript/main.tex` |
| Corresponding author details | Resolved | Authors | Official email and corresponding-author designation | `paper/manuscript/main.tex` |
| Funding statement | Resolved | Authors | Confirmed no-funding statement | `paper/manuscript/main.tex` |
| Acknowledgments | Pending | Authors | Non-funding acknowledgments approved for publication | `paper/manuscript/main.tex` |
| Competing-interest declaration | Resolved | Authors | Journal-ready declaration text | `paper/manuscript/main.tex` |
| CRediT roles | Resolved | Authors | Final contribution breakdown by author | `paper/manuscript/main.tex` |

Resolved metadata now reflected in the draft:

- Authors: `Changrui Zhang`; `Chengwei Yang`
- Affiliation: `Beijing Institute of Technology`
- Corresponding author: `Chengwei Yang`
- Email: `yangchengwei@bit.edu.cn`
- Funding: `This research received no external funding.`

## Priority 2: Experimental metadata

| Item | Status | Who should supply it | Required source material | Update targets |
| --- | --- | --- | --- | --- |
| Exact local LLM backend used for the canonical result suites | Resolved | Experiment owner | Confirmed model identifier | `paper/manuscript/sections/experimental_setup.tex`, `paper/manuscript/sections/discussion.tex` |
| Runtime environment name | Resolved | Experiment owner | Confirmed environment name | `paper/manuscript/sections/experimental_setup.tex`, `paper/manuscript/sections/discussion.tex` |
| Hardware/runtime context | Partially resolved | Experiment owner | CPU/GPU confirmed; RAM and OS notes remain optional if the journal version needs them | `paper/manuscript/sections/experimental_setup.tex` |
| Confirmation that the selected canonical suites are the final ones to submit | Resolved | Authors / experiment owner | Confirmed result-directory scope | `paper/manuscript/notes/evidence_inventory.md`, `paper/manuscript/sections/results.tex` |

Resolved experimental metadata now reflected in the draft:

- Backend model: `deepseek-r1-0528-qwen3-8b`
- Python virtual environment: `Memento`
- CPU: `13th Gen Intel(R) Core(TM) i7-13700F`
- GPU: `NVIDIA GeForce RTX 3060`

## Priority 3: Figures and visual submission assets

| Item | Status | Who should supply it | Required source material | Update targets |
| --- | --- | --- | --- | --- |
| Main-body figures with verified provenance | Resolved for current draft | Authors / experiment owner | Repository-local JSON plus figure-generation script | `paper/manuscript/scripts/generate_paper_figures.py`, `paper/manuscript/figures/`, section files |
| Graphical abstract | Pending | Authors | A validated visual concept and source asset | future asset file, manuscript packaging |
| Figure source confirmation | Resolved for current draft | Authors / experiment owner | Figure script plus source-result mapping | `paper/manuscript/notes/evidence_inventory.md` |

Current figure set in the manuscript:

- `generalization_scene_group_bar.png`: mission-success bar chart for the four settings across the five selected scenarios in `result/generalization_suite_seed7_v2_sceneout_iter20`
- `full_iter50_average_trends.png`: 50-iteration trend figure for the Full setting, averaged across the five scenarios in `result/generalization_suite_seed7_v2_sceneout_iter50`

Current constraint:

- Keep legacy `Figure/` assets excluded unless their relationship to the selected planning-and-simulation evidence line is proven.

## Priority 4: References

| Item | Status | Who should supply it | Required source material | Update targets |
| --- | --- | --- | --- | --- |
| Verify each BibTeX entry already used in the draft | Pending | Authors / manuscript editor | Repository-root reference list text file plus any canonical bibliographic source the authors trust | `paper/manuscript/references.bib` |
| Add any extra references the authors want emphasized | Pending | Authors | Candidate entries already present in the repository-local reference list | `paper/manuscript/references.bib`, section files |

Reference boundary:

- The current draft is local-only and should not add external references unless the project scope explicitly changes.

## Priority 5: LaTeX build environment

Current status:

- Workspace-local TinyTeX installed under `.tools/tinytex/dist/TinyTeX`
- `latexmk` available
- `pdflatex` available
- `bibtex` available
- `biber` available
- `xelatex` available
- `elsarticle.cls` generated from `paper/template_elsarticle/elsarticle/elsarticle.ins`
- `paper/manuscript/main.tex` compiled successfully to `paper/manuscript/main.pdf` on 2026-05-16

Recommended follow-up:

1. Review the generated PDF for line breaks, table placement, and any remaining overfull boxes worth polishing
2. Finalize the competing-interest declaration and CRediT statement
3. Rebuild `paper/manuscript/main.tex` after those editorial updates

## Priority 6: VSCode writing support

Current status:

- Installed: `james-yu.latex-workshop`
- Not detected: `ltex-plus.vscode-ltex-plus`
- Not detected: `streetsidesoftware.code-spell-checker`
- Attempted `code --install-extension ltex-plus.vscode-ltex-plus`
  and `code --install-extension streetsidesoftware.code-spell-checker`
  on 2026-05-15; both failed with `EPERM` while creating
  `AppData\\Roaming\\Code\\logs\\...` and a follow-up
  `ECONNREFUSED 127.0.0.1:9`

Manual fallback:

1. Open VSCode
2. Go to Extensions
3. Search `LTeX+`
4. Search `Code Spell Checker`
5. Install them interactively if local permissions or network policy allow it

## Hard boundaries for the manuscript editor

- Do not add unsupported formal claims such as theorem-backed convergence, Lyapunov guarantees, or a verified PCGrad implementation unless code and citable evidence are both added.
- Do not invent RAM, operating-system details, or graphical-abstract provenance.
- Keep quantitative claims tied to the selected evidence lines documented in `paper/manuscript/notes/evidence_inventory.md`.
