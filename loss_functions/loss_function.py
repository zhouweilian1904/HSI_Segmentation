import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.nn.modules.loss import _Loss


class ContextualSegmentationLoss(nn.Module):
    """
    Loss function that considers both labeled pixels and their contextual background
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(ContextualSegmentationLoss, self).__init__()
        self.alpha = alpha  # weight for supervised loss
        self.beta = beta  # weight for contextual learning
        self.smooth = smooth

    def forward(self, predictions, targets, features):
        """
        Args:
            predictions: [B, C, H, W] class predictions
            targets: [B, H, W] ground truth labels (0 for background)
            features: [B, F, H, W] feature maps from intermediate layer
        """
        batch_size = predictions.size(0)

        # 1. Standard supervised loss for labeled pixels
        labeled_mask = (targets > 0)

        # Calculate IoU loss for labeled pixels
        supervised_loss = self.calculate_iou_loss(predictions, targets, labeled_mask)

        # 2. Contextual consistency loss
        contextual_loss = self.calculate_contextual_loss(features, labeled_mask)

        # Combine losses
        total_loss = self.alpha * supervised_loss + self.beta * contextual_loss

        return total_loss, supervised_loss, contextual_loss

    def calculate_iou_loss(self, predictions, targets, labeled_mask):
        """Calculate IoU loss for labeled pixels"""
        # Standard IoU calculation for labeled pixels
        predictions = predictions * labeled_mask.unsqueeze(1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot * labeled_mask.unsqueeze(1)

        intersection = (predictions * targets_one_hot).sum()
        union = predictions.sum() + targets_one_hot.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

    def calculate_contextual_loss(self, features, labeled_mask):
        """
        Calculate contextual consistency between labeled and background regions
        """
        B, F, H, W = features.shape

        # Get features for labeled and background regions
        labeled_features = features * labeled_mask.unsqueeze(1)
        background_features = features * (~labeled_mask).unsqueeze(1)

        # Calculate feature statistics
        labeled_mean = labeled_features.mean(dim=(2, 3), keepdim=True)
        background_mean = background_features.mean(dim=(2, 3), keepdim=True)

        # Calculate feature similarity using cosine similarity
        labeled_norm = torch.norm(labeled_mean, dim=1, keepdim=True)
        background_norm = torch.norm(background_mean, dim=1, keepdim=True)

        # Avoid division by zero
        labeled_norm = torch.clamp(labeled_norm, min=self.smooth)
        background_norm = torch.clamp(background_norm, min=self.smooth)

        # Normalize features
        labeled_normalized = labeled_mean / labeled_norm
        background_normalized = background_mean / background_norm

        # Calculate cosine similarity
        similarity = (labeled_normalized * background_normalized).sum(dim=1).mean()

        return 1 - similarity

class IoULoss(nn.Module):
    """
    Implementation of IoU (Intersection over Union) loss for patch-based segmentation.
    """

    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape [B, C, H, W] (after softmax)
            targets: Tensor of shape [B, H, W] with class indices
        Returns:
            IoU loss
        """
        # Ensure predictions are in the right format
        if predictions.dim() != 4:
            raise ValueError("Predictions should be of shape [B, C, H, W]")

        # Get dimensions
        batch_size, num_classes, height, width = predictions.size()

        # Reshape predictions and targets for calculation
        predictions = predictions.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        predictions = predictions.view(-1, num_classes)  # [B*H*W, C]

        # Convert targets to one-hot encoding
        targets = targets.view(-1)  # [B*H*W]
        targets = F.one_hot(targets, num_classes=num_classes)  # [B*H*W, C]

        # Calculate intersection and union
        intersection = (predictions * targets).sum(dim=0)  # [C]
        union = predictions.sum(dim=0) + targets.sum(dim=0) - intersection  # [C]

        # Calculate IoU for each class
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)  # [C]

        # Average IoU across classes
        iou = iou_per_class.mean()

        return 1 - iou

    def calculate_iou_scores(self, predictions, targets):
        """
        Calculate IoU scores for each class
        Args:
            predictions: Tensor of shape [B, C, H, W]
            targets: Tensor of shape [B, H, W]
        Returns:
            List of IoU scores for each class
        """
        batch_size, num_classes, height, width = predictions.size()

        # Reshape predictions and targets
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        predictions = predictions.view(-1, num_classes)
        targets = targets.view(-1)
        targets = F.one_hot(targets, num_classes=num_classes)

        # Calculate IoU for each class
        intersection = (predictions * targets).sum(dim=0)
        union = predictions.sum(dim=0) + targets.sum(dim=0) - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)

        return iou_per_class.cpu().detach().numpy()

# Example of contrastive loss
def contrastive_loss(features, labels, margin=1.0):
    batch_size = features.size(0)

    # Compute pairwise distances (L2 norm)
    dist_matrix = torch.cdist(features, features, p=2)

    # Create labels matrix: True for pairs with the same label, False otherwise
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

    # Extract positive pairs and negative pairs
    pos_pairs = dist_matrix[labels_matrix]
    neg_pairs = dist_matrix[~labels_matrix]

    # Calculate losses
    pos_loss = pos_pairs.mean()
    neg_loss = torch.clamp(margin - neg_pairs, min=0).mean()

    return pos_loss + neg_loss


# Example of center loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return torch.mean((features - centers_batch) ** 2)


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[0], target)
        for i in range(1, len(output)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(output, target)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1).float()

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def multi_class_dice_loss(pred, target, num_classes, epsilon=1e-6):
    """
    Compute multi-class Dice loss.

    Args:
    - pred: predicted segmentation logits (batch_size, num_classes, height, width)
    - target: ground truth segmentation labels (batch_size, height, width)
    - num_classes: number of classes

    Returns:
    - dice_loss: the average Dice loss across all classes
    """
    dice_loss = 0.0
    # Loop through each class to calculate dice loss
    for c in range(num_classes):
        pred_c = pred[:, c, :, :]  # Get predictions for class c (batch_size, height, width)
        target_c = (target == c).float()  # Get binary ground truth mask for class c

        # Calculate intersection and union
        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Sum over spatial dimensions
        pred_sum = pred_c.sum(dim=(1, 2))
        target_sum = target_c.sum(dim=(1, 2))

        # Compute dice score
        dice_score = (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)

        # Dice loss is 1 - dice score
        dice_loss += 1 - dice_score.mean()

    # Average the loss across all classes
    dice_loss /= num_classes
    return dice_loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=0):
        """
        Multi-class Dice Loss with support for excluding specific labels (e.g., background).

        Args:
            smooth (float): Smoothing term to prevent division by zero.
            ignore_index (int): Index of the class to ignore (e.g., background class).
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions with shape (B, C, H, W), where C is the number of classes.
            target (Tensor): Ground truth with shape (B, H, W) containing class indices.
        """
        # Apply softmax to predictions for multi-class probabilities
        pred = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Exclude the ignored index from computation
        if self.ignore_index is not None:
            pred = pred[:, 1:, ...]  # Skip the first channel (label=0)
            target = target[:, 1:, ...]  # Skip the first channel (label=0)

        # Flatten the spatial dimensions
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # (B, C, H*W)
        target_flat = target.view(target.shape[0], target.shape[1], -1)  # (B, C, H*W)

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)  # Sum over spatial dimensions
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # Sum over spatial dimensions

        # Compute Dice coefficient for each class
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Average Dice coefficient over classes and batches
        dice_loss = 1 - dice_coeff.mean()

        return dice_loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=False, aux_weight=0.4, ignore_index=0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(output[0], target_one_hot, valid_mask)
        for i in range(1, len(output)):
            aux_loss = self._base_forward(output[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            valid_mask = (target != self.ignore_index).long()
            target_one_hot = F.one_hot(torch.clamp_min(target, 0))
            return self._base_forward(output, target_one_hot, valid_mask)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


class BCELossBoud(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log((1 - output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1 - target) * torch.log((1 - output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        target_one_hot = F.one_hot(torch.clamp_min(target, 0), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        predict = torch.softmax(predict, 1)

        bs, category, depth, width, heigt = target_one_hot.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:, i]
            targ_i = target_one_hot[:, i]
            tt = np.log(depth * width * heigt / (target_one_hot[:, i].cpu().data.numpy().sum() + 1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
            torch.log(torch.mul(std, std))) - 1


class DiceCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss(aux=False)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Cross-Entropy Loss
        ce_loss = self.ce_loss(pred, target)

        # Weighted sum of Dice and Cross-Entropy Losses
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


def segmentation_loss(loss='CE', aux=False, **kwargs):
    if loss == 'dice' or loss == 'DICE':
        seg_loss = DiceLoss(aux=aux)
    elif loss == 'dice-ce' or loss == 'DICE-CE':
        seg_loss = DiceCELoss(aux=aux)
    elif loss == 'crossentropy' or loss == 'CE':
        seg_loss = MixSoftmaxCrossEntropyLoss(aux=aux)
    elif loss == 'bce':
        seg_loss = nn.BCELoss(size_average=True)
    elif loss == 'bcebound':
        seg_loss = BCELossBoud(num_classes=kwargs['num_classes'])
    else:
        print('sorry, the loss you input is not supported yet')
        sys.exit()

    return seg_loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multitask loss

    Params：
        num: int，the number of loss
        x: multitask loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def cosine_similarity_loss(x1, x2):
    cos = F.cosine_similarity(x1, x2, dim=1)
    return 1 - cos.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = F.pairwise_distance(output1, output2, p=2)
        losses = 0.5 * (label * torch.pow(distances, 2) +
                        (1 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        return losses.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class DiceCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Cross-Entropy Loss
        ce_loss = self.ce_loss(pred, target)

        # Weighted sum of Dice and Cross-Entropy Losses
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


def kl_divergence_loss(p, q):
    return F.kl_div(p.log(), q, reduction='batchmean')


import kornia

ssim_loss = kornia.losses.SSIMLoss(window_size=3)


def compute_ssim_loss(output, target):
    return ssim_loss(output, target)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tsne_on_original_data(data, labels, output_dir="classification_maps", perplexity=30, n_iter=1000):
    """
    Perform t-SNE on the data and visualize the results.

    Args:
        data: Hyperspectral image data, shape (H, W, C).
        labels: Ground truth labels, shape (H, W).
        output_dir: Directory to save the plot.
        perplexity: Perplexity parameter for t-SNE.
        n_iter: Number of iterations for t-SNE optimization.
    """
    # Flatten data and labels
    H, W, C = data.shape
    data_flattened = data.reshape(-1, C)  # Shape: (H*W, C)
    labels_flattened = labels.flatten()  # Shape: (H*W,)

    # Remove pixels with label 0 (unlabeled pixels)
    valid_mask = labels_flattened > 0
    data_valid = data_flattened[valid_mask]
    labels_valid = labels_flattened[valid_mask]

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    data_tsne = tsne.fit_transform(data_valid)

    # Plot t-SNE results
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_valid, cmap='jet', edgecolors='white')
    plt.colorbar(scatter, ticks=np.unique(labels_valid))
    plt.title('t-SNE Visualization of Test Features')
    plt.grid()
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "TSNE_on_original_data.png"), dpi=600, bbox_inches="tight")
    plt.close()

    print(f"t-SNE visualization saved to {os.path.join(output_dir, 'TSNE_on_original_data.png')}")
