# Validation performed during portfolio preparation

- Python AST syntax checks for all public scripts.
- Current environment: Python 3.10.8, NumPy 2.2.6, opencv-python 4.13.0.92,
  openpyxl 3.1.5; both original entry scripts' --help paths executed.
- Created two synthetic 8-frame, 3-band, 6-view MP4 fixtures and a 17-column
  annotation workbook. Ran video prediction -> human conversion -> submission
  merge -> comparison CLI. Both synthetic records matched fixture labels.
  This is a mechanical check, not a driving-safety experiment.
- Rebuilt all 100 human JSON records from the private workbook: exact equality
  with saved human submission. Verified merged human and automatic records
  against the two saved source JSON files, all 100 unique IDs, permitted scores,
  means/thresholds, and recomputed all public aggregate metrics.
- Re-evaluated one private course video with the original Rule2 implementation
  and compared its complete output to the corresponding saved record. Input
  stayed local; no original footage was published.
- Relative README/document links, staged file list, synthetic-only data
  exceptions, sizes and targeted credential/private-path patterns checked.

Not verified: rerunning the entire 100-video automatic evaluation, historical
package versions, a fresh dependency installation, threshold calibration,
unseen datasets, independent holdout performance or traffic-object detection.
Course data rights remain unconfirmed; raw course records are not published.
