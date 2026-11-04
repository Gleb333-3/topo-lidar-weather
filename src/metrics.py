# Scores for a noise filter. "Positive" = the point is snow.

import numpy as np


def scores(is_snow, predicted_noise):
    tp = np.sum(predicted_noise & is_snow)      # snow we removed (good)
    fp = np.sum(predicted_noise & ~is_snow)     # real points we removed (bad)
    fn = np.sum(~predicted_noise & is_snow)     # snow we missed (bad)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    kept_real = 1 - fp / max(np.sum(~is_snow), 1)  # share of real points we kept
    return {"precision": precision, "recall": recall, "f1": f1, "kept_real": kept_real}