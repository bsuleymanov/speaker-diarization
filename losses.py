import torch
from torch import nn
import torch.nn.functional as F


def sdep_loss(student_probs, teacher_probs, config=None):
    # student_probs: batch_size x n_networks x n_local_views x n_prototypes
    # teacher_probs: batch_size x n_networks x n_global_views x n_prototypes
    _, _, n_global_views, _ = teacher_probs.shape
    tau_w = config["tau_w"] if config is not None else 1.
    ensemble_weights_per_network = get_loss_for_each_network(
        entropy_loss, teacher_probs, reduction="sum")
    ensemble_weights_per_network = nn.Softmax()(ensemble_weights_per_network)
    ce_loss_per_network = get_loss_for_each_network(
        cross_entropy_loss, student_probs, teacher_probs, reduction="mean")
    ce_loss = torch.mean(ce_loss_per_network * ensemble_weights_per_network)
    reg_loss = entropy_loss(student_probs)
    return ce_loss + reg_loss


def get_loss_for_each_network(loss_fn, pred, target=None, reduction="mean"):
    res = []
    for i in range(pred.shape[1]):
        if target is None:
            res.append(loss_fn(pred[:, i, ...], reduction=reduction))
        else:
            res.append(loss_fn(pred[:, i, ...], target[:, i, ...], reduction=reduction))
    return torch.stack(res)


def cross_entropy_loss(pred, target):
    batch_size, n_local_views, n_prototypes = pred.shape
    _, n_global_views, _ = target.shape
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    n_views = 0
    for target_view_id in range(n_global_views):
        for pred_view_id in range(n_local_views):
            partial_loss = loss_fn(
                pred[:, pred_view_id, ...],
                target[:, target_view_id, ...],
            )
            total_loss += partial_loss
            n_views += 1
    total_loss /= n_views * batch_size
    return total_loss


def kl_div_loss(pred, target):
    batch_size, n_local_views, n_prototypes = pred.shape
    _, n_global_views, _ = target.shape
    loss_fn = nn.KLDivLoss(reduction="sum", log_target=True)
    total_loss = 0
    n_views = 0
    for target_view_id in range(n_global_views):
        for pred_view_id in range(n_local_views):
            partial_loss = loss_fn(
                pred[:, pred_view_id, ...],
                target[:, target_view_id, ...],
            )
            total_loss += partial_loss
            n_views += 1
    total_loss /= n_views * batch_size
    return total_loss


def entropy_loss(probs, reduction="mean"):
    probs = probs.reshape(-1, probs.shape[-1])
    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    #loss_fn_wo_reduction = nn.CrossEntropyLoss(reduction="none")
    #loss = loss_fn_wo_reduction(probs, probs)
    return loss_fn(probs, probs)
