# SMPT Submission Checklist

## Manuscript package

- [x] Main manuscript PDF compiled successfully: `paper/manuscript/main_smpt_submission.pdf`
- [x] Reference list expanded and cross-references resolved
- [x] Author, affiliation, corresponding-author email, funding, CRediT, and competing-interest statements included
- [x] `Acknowledgments` and `Data availability` written in submission-ready prose

## Figures and tables

- [x] Main-text figures generated from repository-local evidence
- [x] Method overview figure included in the manuscript
- [x] Ablation and generalization figures included in the manuscript
- [x] Table layout tightened for `tab:scenario-overview` and `tab:export-correspondence`
- [ ] Final manual pass on residual table/line-break warnings if a cleaner print layout is required

## Graphical abstract and artwork

- [x] Separate graphical abstract prepared without generative AI
- [x] Exported as `PDF`, high-resolution `PNG`, and `TIFF`
- [x] Grayscale-legible flow layout and Arial font used
- [ ] Final visual check at journal upload resolution and thumbnail scale

## Reproducibility and metadata

- [x] Manuscript states the confirmed model, environment, CPU, and GPU
- [x] New result outputs serialize a top-level `runtime_manifest`
- [x] Smoke result generated with `runtime_manifest`: `result/smpt_runtime_smoke/full_result.json`
- [x] Runtime-manifest tests added and passed
- [ ] Decide whether the archived canonical evidence should be rerun so every reported result also contains the new manifest

## Editorial final pass

- [ ] Confirm whether a public repository or archive link will be disclosed at submission
- [ ] Replace the current data-availability wording if the journal requires a stricter template
- [ ] Add any late acknowledgments only if they are genuinely needed
- [ ] Complete final upload checks in the Elsevier submission portal
