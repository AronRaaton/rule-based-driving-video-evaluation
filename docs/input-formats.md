# Inputs and outputs

Rule2 accepts a directory, processes sorted .mp4 names, and writes a JSON list.
It does not accept frame directories or a single video path. Each frame must
have three equal horizontal bands and six horizontal views in each band.
At most eight frames are sampled by default, with step max(1, total//max_frames).
Unreadable videos return no record. Remainder height is discarded; the sixth
view includes remainder width. Use --max-frames > 0 and the expected layout.

Each output record has video_id, is_poisoned, attack_level, scores
(semantic/logical/decision), final_score and reasoning. The final score is
rounded to four decimals. Poison threshold is >= 0.33; greatest-score ties
favor Decision, then Logic, then Semantic.

The annotation converter reads the first worksheet with data_only=True,
skips two header rows, and drops entirely empty rows. Required cells are:
A numeric video index, E semantic score, I logical score, M decision score,
O attack type, Q reasoning. Video IDs become zero-padded NN.mp4. Formula cells
must have cached numeric values; openpyxl does not calculate formulas. Missing
dimension values default to zero. The converter does not validate the five
allowed score values or duplicate IDs.

The converter recalculates final_score and poisoned status from scores. It
keeps an explicit attack type for poisoned records, unless None, and sets
negative records to None in its current implementation. It joins automated
records on video_id and writes human-only and merged JSON lists. Missing auto
files warn and yield null automated_evaluation, so check coverage before use.
Comparison reports only matched IDs.

The synthetic demo script creates all necessary inputs. Its JSON includes
the exact same schema as course output; its directory name and documentation
explicitly distinguish synthetic fixtures from real course evaluation.
