import torch
from abc import ABC, abstractmethod


__all__ = ["InfoNCELoss", "MarginMSELoss", "KLDLoss", "FlopsLoss", "L1Loss", "L0Loss"]


class InfoNCELoss(torch.nn.Module):
    """
    Information Noise Contrastive Estimation (InfoNCE) loss. Reference: https://arxiv.org/abs/1807.03748
    
    :param temperature: Temperature parameter that scales the logits.
    """
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        :param pos_scores: A tensor of shape (batch_size) containing the positive scores.
        :param neg_scores: A tensor of shape (batch_size, n_negatives) containing the negative scores.
        :returns: The loss value.
        """
        scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1) / self.temperature
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device) # Positive to match is always at position 0.
        return self.cross_entropy(scores, labels)


class MarginMSELoss(torch.nn.Module):
    """
    Margin Mean Squared Error (Margin-MSE) loss. Reference: https://arxiv.org/abs/2010.02666
    """
    def __init__(self, teacher_scale: float = 1.0):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.teacher_scale = teacher_scale

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, teacher_pos_scores: torch.Tensor, teacher_neg_scores: torch.Tensor) -> torch.Tensor:
        """
        :param pos_scores: A tensor of shape (batch_size) containing the scores from the student model for positive passages.
        :param neg_scores: A tensor of shape (batch_size, n_negatives) containing the scores from the student model for negative passages.
        :param teacher_pos_scores: A tensor of shape (batch_size) containing the scores from the teacher model for positive passages.
        :param teacher_neg_scores: A tensor of shape (batch_size, n_negatives) containing the scores from the teacher model for negative passages.
        :returns: The loss value.
        """
        pos_scores = pos_scores.unsqueeze(-1)
        teacher_pos_scores = teacher_pos_scores.unsqueeze(-1)
        student_margin = pos_scores - neg_scores
        teacher_margin = (teacher_pos_scores - teacher_neg_scores) * self.teacher_scale
        return self.loss(student_margin, teacher_margin)


class KLDLoss(torch.nn.Module):
    """
    Kullback-Leibler Divergence (KLD) loss. Reference: https://arxiv.org/abs/2010.11386
    """
    def __init__(self, teacher_scale: float = 1.0):
        super().__init__()
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.teacher_scale = teacher_scale

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, teacher_pos_scores: torch.Tensor, teacher_neg_scores: torch.Tensor) -> torch.Tensor:
        """
        :param pos_scores: A tensor of shape (batch_size) containing the scores from the student model for positive passages.
        :param neg_scores: A tensor of shape (batch_size, n_negatives) containing the scores from the student model for negative passages.
        :param teacher_pos_scores: A tensor of shape (batch_size) containing the scores from the teacher model for positive passages.
        :param teacher_neg_scores: A tensor of shape (batch_size, n_negatives) containing the scores from the teacher model for negative passages.
        :returns: The loss value.
        """
        student_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=1)
        teacher_scores = torch.cat([teacher_pos_scores.unsqueeze(-1), teacher_neg_scores], dim=1) * self.teacher_scale
        student_log_probs = torch.log_softmax(student_scores, dim=1)
        teacher_probs = torch.softmax(teacher_scores, dim=1)
        return self.loss(student_log_probs, teacher_probs)


class RegularizationLoss(ABC, torch.nn.Module):
    """
    Abstract class for regularization losses.

    :param weight: The target weight of the regularization term.
    """
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, reps: torch.Tensor, **kwargs) -> torch.Tensor:
        """ 
        Computes the weighted regularization loss.

        :param reps: A tensor of shape [batch_size, embeddding_dim] containing the text representations.
        :param kwargs: Additional arguments to be passed to the weight function.
        :returns: A tensor of shape [1] containing the regularization loss.
        """
        return self.reg(reps) * self.w(**kwargs)

    @abstractmethod
    def reg(self, reps: torch.Tensor) -> torch.Tensor:
        """ Computes the regularization term. """
        pass

    @abstractmethod
    def w(self, **kwargs) -> float:
        """ Get the weight of the regularization term. """
        pass


class FlopsLoss(RegularizationLoss):
    """
    Weighted FLOPS regularization loss. Reference: https://arxiv.org/abs/2004.05665
    The weight increases quadratically until a given step, after which it remains constant.
    This is to mitigate the contribution of the FLOPS regularization at the early stages of training.

    :param weight: The target weight of the regularization term.
    :param target_step: The step at which the weight of the regularization term should stop increasing.
    """
    def __init__(self, weight: float, target_step: int = None):
        super().__init__(weight)
        self.target_step = target_step

    def reg(self, reps: torch.Tensor) -> torch.Tensor:
        """ Computes the FLOPS regularization term. """
        return torch.sum(torch.mean(torch.abs(reps), dim=0) ** 2, dim=0)

    def w(self, step: int = None) -> float:
        """ Computes the weight of the FLOPS regularization term. """
        if self.target_step is not None and step is not None and step < self.target_step:
            return min(self.weight, self.weight * ((step) / (self.target_step+1)) ** 2)
        else:
            return self.weight


class L1Loss(RegularizationLoss):
    """
    L1 regularization loss.
    """
    def __init__(self, weight: float):
        super().__init__(weight)

    def reg(self, reps: torch.Tensor) -> torch.Tensor:
        """ Computes the L1 regularization term. """
        return torch.sum(torch.abs(reps), dim=-1).mean()

    def w(self, **kwargs) -> float:
        """ Computes the weight of the L1 regularization term. """
        return self.weight


class L0Loss(RegularizationLoss):
    """
    L0 regularization loss.
    """
    def __init__(self, weight: float):
        super().__init__(weight)

    def reg(self, reps: torch.Tensor) -> torch.Tensor:
        """ Computes the L0 regularization term. """
        return torch.count_nonzero(reps, dim=-1).float().mean()

    def w(self, **kwargs) -> float:
        """ Computes the weight of the L0 regularization term. """
        return self.weight
