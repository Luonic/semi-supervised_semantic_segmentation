import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lovasz


class CalculateLoss():
    def __init__(self, losses):
        # losses: dict of:
        # 'loss_fn': class instance or fn to run like loss_fn(prediction, target)
        # 'weight': scalar to multiply loss to
        self.losses = losses

    def __call__(self, predictions_list, target):
        loss = 0
        for prediction_idx, prediction in enumerate(predictions_list):
            prediction = torch.nn.functional.interpolate(prediction, size=(target.size(2), target.size(3)),
                                                         mode='bilinear', align_corners=False)
            for loss_spec in self.losses:
                loss += loss_spec['loss_fn'](prediction, target) * loss_spec['weight'][prediction_idx]
        return loss


class DenseCrossEntropyLossWithLogits(nn.Module):
    def __init__(self, reduction='mean'):
        super(DenseCrossEntropyLossWithLogits, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = torch.log_softmax(input, dim=1)
        loss = -(target * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'incorrect value of reduction param: `{self.reduction}`')
        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, thres=0.7, min_kept=100000):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.criterion = DenseCrossEntropyLossWithLogits(reduction='none')

    def _ohem_forward(self, logits, target, **kwargs):
        pred = F.softmax(logits, dim=1)
        pixel_losses = self.criterion(logits, target).contiguous().view(-1)

        pred, ind = pred.contiguous().view(-1, ).contiguous().sort()
        print('calculating min value')
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        
        print('rearrange pixel losses')
        print(ind.shape, pixel_losses.shape)
        pixel_losses = pixel_losses[ind]
        print('get valid losses')
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        return self._ohem_forward(score, target)


class FocalLoss(nn.Module):
    # TODO: Improve numerical stability by clipping sigmoid
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        prediction_probabilities = torch.sigmoid(inputs)
        p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = (targets * self.alpha + (1 - targets) * (1 - self.alpha))
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent

        # normalizer = modulating_factor.sum().detach()
        # return focal_cross_entropy_loss.sum() / (normalizer + 0.001)

        return focal_cross_entropy_loss.mean()


class NormalizedFocalLossSigmoid(nn.Module):

    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0
        t = torch.ones_like(one_hot)

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        pt = torch.where(one_hot, pred, 1 - pred)
        pt = torch.where(label != self._ignore_label, pt, torch.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        t_sum = torch.sum(t, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult

        ignore_area = torch.sum(label == -1, dim=tuple(range(1, label.dim()))).cpu().numpy()
        sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        sample_weight = label != self._ignore_label

        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


# class FocalLoss(nn.Module):
#     def __init__(self, axis=-1, alpha=0.25, gamma=2,
#                  from_logits=False, batch_axis=0,
#                  weight=None, num_class=None,
#                  eps=1e-9, size_average=True, scale=1.0):
#         super(FocalLoss, self).__init__()
#         self._axis = axis
#         self._alpha = alpha
#         self._gamma = gamma
#         self._weight = weight if weight is not None else 1.0
#         self._batch_axis = batch_axis
#
#         self._scale = scale
#         self._num_class = num_class
#         self._from_logits = from_logits
#         self._eps = eps
#         self._size_average = size_average
#
#     def forward(self, pred, label, sample_weight=None):
#         if not self._from_logits:
#             pred = F.sigmoid(pred)
#
#         one_hot = label > 0
#         pt = torch.where(one_hot, pred, 1 - pred)
#
#         t = label != -1
#         alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
#         beta = (1 - pt) ** self._gamma
#
#         loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
#         sample_weight = label != -1
#
#         loss = self._weight * (loss * sample_weight)
#
#         if self._size_average:
#             tsum = torch.sum(label == 1, dim=get_dims_with_exclusion(label.dim(), self._batch_axis))
#             loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
#         else:
#             loss = torch.sum(loss, dim=get_dims_with_exclusion(loss.dim(), self._batch_axis))
#
#         return self._scale * loss


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def dice_loss(input, target):
    smooth = 1.

    intersection = (input * target).sum(dim=(1, 2, 3))
    cardinality = (input + target).sum(dim=(1, 2, 3))

    return 1 - ((2. * intersection + smooth) / (cardinality + smooth))


def log_dice_loss(input, target, gamma=1.0):
    smooth = 1.

    intersection = (input * target).sum(dim=(1, 2, 3))
    cardinality = (input + target).sum(dim=(1, 2, 3))

    return torch.pow(-torch.log(((2. * intersection + smooth) / (cardinality + smooth))), exponent=gamma)


def binary_lovasz_loss_with_logits(input, target):
    int_target = torch.argmax(target, dim=1, keepdim=False)
    # input = torch.sigmoid(input)
    target_list = torch.split(int_target, 1, dim=0)
    input_list = torch.split(input, 1, dim=0)
    loss = 0
    num_valid_samples = 0
    for inp, tgt in zip(input_list, target_list):
        mask_sample = (tgt.sum() > 0).to(input)
        num_valid_samples += mask_sample
        loss += lovasz.lovasz_softmax(inp, tgt, classes=[1], ignore=255, per_image=True) * mask_sample
    return loss / (num_valid_samples + 0.001)


def binary_entropy_loss(input):
    input = torch.sigmoid(input)
    entropy_pos = input * torch.log(input)
    entropy_neg = (1. - input) * torch.log(1. - input)
    return -torch.mean(entropy_pos + entropy_neg) / 2


def entropy_loss(input):
    prob = torch.softmax(input, dim=1)
    entropy = prob * torch.log_softmax(input, dim=1)
    loss = -torch.sum(entropy, dim=1)
    return loss.mean()


def smooth_binary_labels(target, alpha=0.1):
    return target * (1 - alpha) + alpha / 2
