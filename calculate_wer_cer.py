"""
A standalone script that compares a folder of predicted transcripts
against a folder of ground truth transcripts and reports the overall
metrics
"""

import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def main():
    """
    Reads the predicted and ground truth files, matches them up by name,
    and prints the average metrics
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_dir",
        required=True,
        type=str,
        help="Directory of predicted .txt files",
    )
    parser.add_argument(
        "--ground_truth_dir",
        required=True,
        type=str,
        help="Directory of ground-truth .txt files",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    gt_dir = Path(args.ground_truth_dir)

    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}

    common_ids = sorted(set(pred_files) & set(gt_files))
    missing_pred = sorted(set(gt_files) - set(pred_files))
    missing_gt = sorted(set(pred_files) - set(gt_files))

    if missing_pred:
        preview = missing_pred[:5]
        suffix = "..." if len(missing_pred) > 5 else ""
        print(
            f"Warning: {len(missing_pred)} ground-truth files have no "
            f"matching prediction: {preview}{suffix}"
        )
    if missing_gt:
        preview = missing_gt[:5]
        suffix = "..." if len(missing_gt) > 5 else ""
        print(
            f"Warning: {len(missing_gt)} predictions have no matching "
            f"ground truth: {preview}{suffix}"
        )

    if not common_ids:
        print("No matching prediction/ground-truth pairs found")
        return

    cers, wers = [], []
    for uid in common_ids:
        target = CTCTextEncoder.normalize_text(gt_files[uid].read_text())
        pred = CTCTextEncoder.normalize_text(pred_files[uid].read_text())
        cers.append(calc_cer(target, pred))
        wers.append(calc_wer(target, pred))

    print(f"\nEvaluated {len(common_ids)} files")
    print(f"CER: {sum(cers) / len(cers):.4f}")
    print(f"WER: {sum(wers) / len(wers):.4f}")


if __name__ == "__main__":
    main()
