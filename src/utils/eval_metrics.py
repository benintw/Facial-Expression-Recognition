"""
Evaluation metrics for facial expression recognition.

This module implements various metrics for model evaluation:
- Multiclass classification accuracy
- Additional metrics can be added as needed
"""

import torch


def get_multiclass_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy for multiclass classification predictions.

    Args:
        y_pred: Model predictions as a tensor of shape (N, num_classes)
        y_true: Ground truth labels as a tensor of shape (N,)

    Returns:
        torch.Tensor: The accuracy as a value between 0 and 1
    """
    top_p, top_class = y_pred.topk(k=1, dim=1)
    corrects = top_class == y_true.reshape(top_class.shape)
    return torch.mean(corrects.type(torch.float32))
