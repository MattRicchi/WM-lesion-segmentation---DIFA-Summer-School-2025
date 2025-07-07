import numpy as np

def evaluate_segmentation(pred_mask, gt_mask):
    """
    Evaluate segmentation performance.

    Args:
        pred_mask (np.ndarray): Binary predicted mask (0 or 1).
        gt_mask (np.ndarray): Binary ground truth mask (0 or 1).

    Returns:
        dict: Dictionary with TPF, FPF, and Dice coefficient.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # True Positives (TP): predicted 1, ground truth 1
    TP = np.logical_and(pred_mask, gt_mask).sum()
    # False Positives (FP): predicted 1, ground truth 0
    FP = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    # False Negatives (FN): predicted 0, ground truth 1
    FN = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    # True Negatives (TN): predicted 0, ground truth 0
    TN = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()

    # True Positive Fraction (Sensitivity, Recall)
    TPF = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # False Positive Fraction
    FPF = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    # Dice Similarity Coefficient
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    return {
        'True Positive Fraction': TPF,
        'False Positive Fraction': FPF,
        'Dice Coefficient': dice
    }

# # Example usage:
# # Example masks (replace with your own)
# pred = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
# gt = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
# results = evaluate_segmentation(pred, gt)
# for k, v in results.items():
#     print(f"{k}: {v:.4f}")