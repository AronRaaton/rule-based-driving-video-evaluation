# Rule-Based Safety Evaluation of Generated Driving Videos

## Project overview
A team course project evaluating generated driving videos through semantic,
logical and decision-level error proxies. The selected automated pipeline is
local Python/OpenCV rules, alongside human annotations and JSON submission
assembly. It is not an LLM evaluator.

## My contribution
I was responsible for the Python code. Human annotation, framework/report work
and other team members' contributions are not presented as my independent work.

## Methods and implementation
`Rule2.py` samples up to eight frames per MP4. Each frame has three horizontal
bands (ground truth, 3D box map, generated result), each with six views ordered
Front-Left, Front, Front-Right, Rear-Left, Rear, Rear-Right.

- Semantic proxy: grayscale/edge mismatch in center and lower regions, blur and
  box-map layout mismatch; combines weighted views and upper percentiles.
- Logical proxy: temporal pixel/edge-motion gaps, jumps and jitter.
- Decision proxy: front-view lower-road mismatch, box-edge density, motion gaps
  and a stationary-reference/moving-generation heuristic.

Rule evidence is quantized to 0, 0.25, 0.5, 0.75 or 1. Final score is the mean
of the three dimensions; `is_poisoned` is true at `final_score >= 0.33`.
Dominant type is the greatest dimension score with tie priority
Decision > Logic > Semantic; clean samples get `None`. This follows the team's
rule design; the course guideline supplies the three-level taxonomy, not the
implementation's numerical thresholds. These signals do not detect named
traffic objects or establish a causal attack. Explanations are templates.

## Repository structure
- `Rule2.py`: selected revised rule evaluator.
- `generate_submission_from_annotations.py`: human XLSX conversion, merge and metrics.
- `examples/`: newly created **synthetic demo** video, workbook and JSON outputs.
- `results/course-comparison.json`: aggregate comparison from existing course results.
- `compare_results.py`: new portable comparison command for user-supplied JSON.
- `docs/`: audit, validation, input formats and publication notes.

## Setup and usage
Verified current environment: Python 3.10.8, NumPy 2.2.6, opencv-python
4.13.0.92 and openpyxl 3.1.5. Historical environment versions were not recorded;
requirements list actual direct imports without pretending to be a lockfile.

```bash
python -m venv .venv
# Activate .venv using your shell's standard command.
python -m pip install -r requirements.txt
python Rule2.py --help
python generate_submission_from_annotations.py --help
python examples/create_synthetic_demo.py
python Rule2.py --input examples/synthetic_demo/videos --output examples/synthetic_demo/auto.json --max-frames 8
python generate_submission_from_annotations.py --xlsx examples/synthetic_demo/annotations.xlsx --auto-json examples/synthetic_demo/auto.json --human-output examples/synthetic_demo/human.json --dataset-output examples/synthetic_demo/submission.json
python compare_results.py --human examples/synthetic_demo/human.json --auto examples/synthetic_demo/auto.json
```

The demo contains simple generated geometric patterns, not real driving footage.
Its scores and manual labels are fixture values for exercising the pipeline,
not course results or a safety benchmark. Input/output details are documented
in [formats](docs/input-formats.md). Obtain original evaluation videos and
annotations from the course team with permission; put them in an ignored local
directory and substitute their paths in the commands above. No external
service or API credential is required.

## Results
Computed during portfolio preparation from the existing human-only and revised
Rule2 JSON files; all 100 IDs match. Both sets have 27 poisoned labels. The
merged submission was checked against both inputs. These are course-sample
agreements against **converter-normalized human labels**, not a held-out score.

| Metric | Value |
| --- | ---: |
| Binary agreement | 70 / 100 (70%) |
| Precision / recall / F1 | 0.444 / 0.444 / 0.444 |
| TP / TN / FP / FN | 12 / 58 / 15 / 15 |
| Attack-level exact agreement (including None) | 61 / 100 (61%) |
| Semantic / logical / decision score MAE | 0.3575 / 0.3100 / 0.1675 |

Raw course annotations, per-video predictions and reasoning are withheld because
redistribution rights are unconfirmed. Only aggregates are public, so the course
comparison requires authorized source files to recalculate. The synthetic
pipeline is independently runnable. No threshold-tuning or independent test
partition record exists; do not infer out-of-sample reliability.

## Limitations
Handcrafted image differences approximate safety concepts without object
detection, tracking or driving-policy reasoning. View weights enter semantic
aggregation twice (per-view scaling and weighted mean), as in the original code.
Sampling skips unreadable files without failing, requires the fixed mosaic
layout, and assumes reasonable image dimensions. Scores depend on thresholds
and rendering; small validation examples do not establish safety validity.

The converter recalculates human poisoned status from dimension scores and
retains an explicit human attack label when available; it does not preserve an
original yes/no label column. Rebuilding from the workbook matches the saved
human JSON exactly; all negative records have type `None`.
See [audit](docs/pipeline-audit.md) and [validation](docs/validation.md).

## References and acknowledgements
The course's Annotation Guideline provides semantic, logical and decision-level
categories and the three-band/six-view input setting. The team's rule design
adds scoring and aggregation. Course documents/data and teammates' reports are
not redistributed. OpenCV, NumPy and openpyxl provide third-party implementations.
`driving_safety_model.py` is an earlier continuous-score rule baseline;
`LLM-base.py` is an alternative API prototype with no matching adopted output.
Both are excluded from this selected showcase, as is the simple counting helper.
No new open-source license was added; see
[publication notes](docs/publication-notes.md).
