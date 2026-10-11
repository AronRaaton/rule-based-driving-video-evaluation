"""New portfolio utility; uses the original converter's comparison formulas."""
import argparse
import json
from generate_submission_from_annotations import print_comparison_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human',required=True)
    parser.add_argument('--auto',required=True)
    args = parser.parse_args()
    with open(args.human,encoding='utf-8') as f:
        human = json.load(f)
    with open(args.auto,encoding='utf-8') as f:
        auto = json.load(f)
    for label, rows in (('human',human),('auto',auto)):
        ids = [r['video_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate video IDs in {label} records')
    if {r['video_id'] for r in human} != {r['video_id'] for r in auto}:
        raise ValueError('Human and automatic video ID sets differ')
    print_comparison_summary(human,{r['video_id']:r for r in auto})


if __name__ == '__main__':
    main()
