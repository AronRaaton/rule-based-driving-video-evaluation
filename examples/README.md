# Synthetic demo

All assets in synthetic_demo are newly generated geometric fixtures. No course
frame, video, annotation, or human reasoning was copied into these assets.
Each MP4 has eight frames, three bands and six views per band. 00.mp4 repeats
matching geometric content; 01.mp4 alternates black/white generated content.
The workbook uses deliberate fixture scores (all zero or all one). Result
agreement is not evidence of safety performance.

Run create_synthetic_demo.py to regenerate videos and workbook. The main README
provides commands that generate auto.json, human.json and submission.json.
The committed JSON files show the actual verified output of that workflow.
