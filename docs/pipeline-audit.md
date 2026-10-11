# Pipeline and historical output audit

The selected chain is Rule2.py -> auto_predictions_rule_v2_revised.json ->
generate_submission_from_annotations.py -> merged dataset JSON. The merged
file's rule design names rule_based_rule2, and all 100 automated records are
exactly equal to the revised prediction records. All 100 human records equal
the human-only JSON. Rebuilding that human JSON from the annotation workbook
with the available converter produced exact equality, including reasoning.

All three files have the same 100 unique video IDs. Score means, poisoned
thresholds, permitted discrete values and type consistency were checked.
The original workbook and record files remain private; this repository publishes
only aggregate metrics. Positive label is is_poisoned=true. Comparison metrics
are computed by video_id, with the converter-normalized human labels as reference.
Binary agreement=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN);
F1=2TP/(2TP+FP+FN). Attack agreement includes None. Score MAE averages absolute
human/automatic differences per dimension. The comparison was recalculated
during portfolio preparation, not claimed as an original experiment log.

The current converter recalculates human poisoned status from dimension scores,
rather than reading an original yes/no field. It preserves supplied attack types
for positives. Consequently its human label logic does not necessarily implement
the automatic greatest-score/tie rule. Current human negatives are all None.
The exact threshold calibration history and independent validation set are absent.

Earlier driving_safety_model.py computes continuous scores and uses max-dimension
threshold 0.2, unlike Rule2's quantization and mean threshold. It is excluded.
auto_predictions.json is an earlier output family, not the merged final input.
LLM-base.py is an alternative API prototype; no adopted matching prediction file
was found. Jduge.py only counts labels; it does not compare with human annotations.
It is excluded in favor of the original converter's actual comparison functions.

The course guideline defines semantic, logical and decision categories and the
input mosaic. Its example contains continuous scores, whereas the team's rule
document and final implementation use five discrete scores. The rule document's
high-level safety criteria are design intent: the code actually computes image
statistics, not verified detections of vehicles, traffic lights or causal attacks.
Rule2's text reasoning is generic templated explanation, not localized evidence.

The available scripts are preserved byte-for-byte. The new demo and comparison
wrapper are portfolio additions. Course results were not edited.
