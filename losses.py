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
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
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


class DiceWithLogitsLoss(nn.Module):

    def forward(self, input, target):
        probs = torch.softmax(input, dim=1)
        smooth = 1.
        target_mask = (target.sum(dim=(2, 3)) > 1.).to(target)
        intersection = (probs * target).sum(dim=(2, 3))
        cardinality = (probs + target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dice = (dice * target_mask).sum(dim=1) / target_mask.sum(dim=1)
        return 1 - dice.mean()


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


def smooth_labels(target, alpha=0.1):
    return target * (1 - alpha) + alpha / target.size(1)


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    _euler_num = 2.718281828  # euler number
    _pi = 3.14159265  # pi
    _ln_2_pi = 1.837877  # ln(2 * pi)
    _CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
    _CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
    _POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
    _IS_SUM = 1  # sum the loss per channel

    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def map_get_pairs(self, labels_4D, probs_4D, radius=3, is_combine=True):
        """get map pairs
        Args:
            labels_4D	:	labels, shape [N, C, H, W]
            probs_4D	:	probabilities, shape [N, C, H, W]
            radius		:	the square radius
        Return:
            tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
        """
        # pad to ensure the following slice operation is valid
        # pad_beg = int(radius // 2)
        # pad_end = radius - pad_beg

        # the original height and width
        label_shape = labels_4D.size()
        h, w = label_shape[2], label_shape[3]
        new_h, new_w = h - (radius - 1), w - (radius - 1)
        # https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
        # padding = (pad_beg, pad_end, pad_beg, pad_end)
        # labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

        # get the neighbors
        la_ns = []
        pr_ns = []
        # for x in range(0, radius, 1):
        for y in range(0, radius, 1):
            for x in range(0, radius, 1):
                la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
                pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            # for calculating RMI
            pair_ns = la_ns + pr_ns
            p_vectors = torch.stack(pair_ns, dim=2)
            return p_vectors
        else:
            # for other purpose
            la_vectors = torch.stack(la_ns, dim=2)
            pr_vectors = torch.stack(pr_ns, dim=2)
            return la_vectors, pr_vectors

    def map_get_pairs_region(self, labels_4D, probs_4D, radius=3, is_combine=0, num_classeses=21):
        """get map pairs
        Args:
            labels_4D	:	labels, shape [N, C, H, W].
            probs_4D	:	probabilities, shape [N, C, H, W].
            radius		:	The side length of the square region.
        Return:
            A tensor with shape [N, C, radiu * radius, H // radius, W // raidius]
        """
        kernel = torch.zeros([num_classeses, 1, radius, radius]).type_as(probs_4D)
        padding = radius // 2
        # get the neighbours
        la_ns = []
        pr_ns = []
        for y in range(0, radius, 1):
            for x in range(0, radius, 1):
                kernel_now = kernel.clone()
                kernel_now[:, :, y, x] = 1.0
                la_now = F.conv2d(labels_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
                pr_now = F.conv2d(probs_4D, kernel_now, stride=radius, padding=padding, groups=num_classeses)
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            # for calculating RMI
            pair_ns = la_ns + pr_ns
            p_vectors = torch.stack(pair_ns, dim=2)
            return p_vectors
        else:
            # for other purpose
            la_vectors = torch.stack(la_ns, dim=2)
            pr_vectors = torch.stack(pr_ns, dim=2)
            return la_vectors, pr_vectors
        return

    def log_det_by_cholesky(self, matrix):
        """
        Args:
            matrix: matrix must be a positive define matrix.
                    shape [N, C, D, D].
        Ref:
            https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
        """
        # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
        # where C is the cholesky decomposition of A.
        chol = torch.cholesky(matrix)
        # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
        return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

    def batch_cholesky_inverse(self, matrix):
        """
        Args: 	matrix, 4-D tensor, [N, C, M, M].
                matrix must be a symmetric positive define matrix.
        """
        chol_low = torch.cholesky(matrix, upper=False)
        chol_low_inv = self.batch_low_tri_inv(chol_low)
        return torch.matmul(chol_low_inv.transpose(-2, -1), chol_low_inv)

    def batch_low_tri_inv(self, L):
        """
        Batched inverse of lower triangular matrices
        Args:
            L :	a lower triangular matrix
        Ref:
            https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing
        """
        n = L.shape[-1]
        invL = torch.zeros_like(L)
        for j in range(0, n):
            invL[..., j, j] = 1.0 / L[..., j, j]
            for i in range(j + 1, n):
                S = 0.0
                for k in range(0, i + 1):
                    S = S - L[..., i, k] * invL[..., k, j].clone()
                invL[..., i, j] = S / L[..., i, i]
        return invL

    def log_det_by_cholesky_test(self):
        """
        test for function log_det_by_cholesky()
        """
        a = torch.randn(1, 4, 4)
        a = torch.matmul(a, a.transpose(2, 1))
        print(a)
        res_1 = torch.logdet(torch.squeeze(a))
        res_2 = self.log_det_by_cholesky(a)
        print(res_1, res_2)

    def batch_inv_test(self):
        """
        test for function batch_cholesky_inverse()
        """
        a = torch.randn(1, 1, 4, 4)
        a = torch.matmul(a, a.transpose(-2, -1))
        print(a)
        res_1 = torch.inverse(a)
        res_2 = self.batch_cholesky_inverse(a)
        print(res_1, '\n', res_2)

    def mean_var_test(self):
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)

        x_mean = x.mean(dim=1, keepdim=True)
        x_sum = x.sum(dim=1, keepdim=True) / 2.0
        y_mean = y.mean(dim=1, keepdim=True)
        y_sum = y.sum(dim=1, keepdim=True) / 2.0

        x_var_1 = torch.matmul(x - x_mean, (x - x_mean).t())
        x_var_2 = torch.matmul(x, x.t()) - torch.matmul(x_sum, x_sum.t())
        xy_cov = torch.matmul(x - x_mean, (y - y_mean).t())
        xy_cov_1 = torch.matmul(x, y.t()) - x_sum.matmul(y_sum.t())

        print(x_var_1)
        print(x_var_2)

        print(xy_cov, '\n', xy_cov_1)

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        # loss = self.forward_softmax_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):
        """
        Using both softmax and sigmoid operations.
        Args:
            logits_4D     :    [N, C, H, W], dtype=float32
            labels_4D     :    [N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=self._CLIP_MIN, max=self._CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D     :    [N, C, H, W], dtype=float32
            labels_4D     :    [N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + self._CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D     :    [N, C, H, W], dtype=float32
            probs_4D     :    [N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = self.map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * self._POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * self.log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * self._POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #    rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if self._IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss
