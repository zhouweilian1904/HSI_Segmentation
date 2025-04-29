import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
import get_cls_map_v3
import time
from tqdm.auto import tqdm
from torchinfo import summary
from torch.utils.data import DataLoader
from loss_functions.loss_function import AutomaticWeightedLoss, MultiClassDiceLoss, compute_ssim_loss, CenterLoss, \
    contrastive_loss
import seaborn as sns
# ----------------------------------------------------------------------------------------
# --------------------------------------HSIseg--------------------------------------------------
from other_models import Mamba2D_v3, DFINet, XNet, XNetv2, RNN_2D, SSFTTnet, X_Net_2D_Mamba, UNet, X_Net_2D_CNN, \
    X_Net_2D_CNN_v2, SwinUnet, TransUnet, Swin_Transformer, Swin_Transformer_v2, YNet, PNet, VNet, SegFormer, UNet3D, \
    general_vit_2, Differential_Transformer, Differential_Transfofmer_UNet, TokenLearner, RegionTransformer_v2, \
    HSIseg_v2, HSIseg_v1, HSIseg_v3, HSIseg_v4, TransHSI, MiM_v1, MiM_v2, MiM_v3, MFT, DeepSFT
from other_models.vit_pytorch import t2t, deepvit, cait, crossformer, mobile_vit, pit, cross_vit, mae, regionvit, \
    local_vit, mae, vit_for_small_dataset, vit

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
NUM_CLASSES = 9  # 'PU dataset 训练的时候别忘记再加上class 0（NUM_CLASS + 1）'
BATCH_SIZE_TRAIN = 64
EPOCH = 200
PCA_NUM = 30
PATCH_SIZE = 11
TRAIN_SAMPLE_RATIO = 0.005
FIXED_NUMBER_PER_CLASS = 10
USE_SAMPLE_TYPE = 'fix'  # (random, fix，exist)
current_time = time.strftime("%Y%m%d_%H%M%S")
RUN_TIMES = 1  #如果RUN_TIMES等于1，则表示supervised learning，如果大于1，则是semi-supervised learning (progressive pesudo labelling)
PROBABILITIC_MASK = 0.8  #用来调整MASK的值去限制pesudo labelling的范围
TOLERANCE = -0.005  #如果设置为0，则表示MASK阈值不变

#### DATA AUGMENTATION #####
# USE_AUGMENTATION = True
# USE_MASK = True
# USE_NOISE = True

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# net = RNN_2D.Net(in_channels=PCA_NUM, hidden_channels=64, output_channels=64, patch_size=PATCH_SIZE,  num_class=NUM_CLASSES + 1)

# net = X_Net_2D_Mamba.UNet(n_channels=PCA_NUM, x_n_channels=1, n_classes=NUM_CLASSES + 1, patch_size=PATCH_SIZE, dim=32, dropout_rate=0.1)

# net = Mamba2D_v3.Mamba2D_v3(channels=PCA_NUM, num_classes=NUM_CLASSES + 1, image_size=PATCH_SIZE, )

# net = X_Net_2D_CNN.UNet(n_channels=PCA_NUM, x_n_channels=1, n_classes=NUM_CLASSES + 1)

# net = X_Net_2D_CNN_v2.UNet(n_channels=PCA_NUM, x_n_channels=1, n_classes=NUM_CLASSES + 1)

# net = DFINet.Net(channel_hsi=PCA_NUM, channel_msi=1, class_num=NUM_CLASSES + 1, type='seg')

# net = MFT.MFT(FM=16, NC=PCA_NUM, NCLidar=1, Classes=NUM_CLASSES + 1, patchsize=PATCH_SIZE, type='seg')

# net = DeepSFT.Proposed(dataset_name='PaviaU', in_channels=PCA_NUM, num_classes=NUM_CLASSES + 1, type='seg')

net = SSFTTnet.SSFTTnet(in_channels=PCA_NUM, num_classes= NUM_CLASSES + 1, patch_size=PATCH_SIZE, type='cls')

# net = TransHSI.TransHSI(patch_size=PATCH_SIZE, num_classes=NUM_CLASSES + 1, type='seg')

# net = MiM_v1.MiM_v1(channels=PCA_NUM, num_classes=NUM_CLASSES + 1, image_size=PATCH_SIZE,)

# net = MiM_v2.MiM_v2(channels=PCA_NUM, num_classes=NUM_CLASSES + 1, image_size=PATCH_SIZE,)

# net = MiM_v3.MiM_v3(channels=PCA_NUM, num_classes=NUM_CLASSES + 1, center_pixel=False, image_size=PATCH_SIZE,)

# net = UNet.UNet(n_channels=PCA_NUM, x_n_channels=1, n_classes=NUM_CLASSES + 1, bilinear=True, type = 'seg')

# net = Nested_UNet.NestedUNet(input_channels=PCA_NUM, num_classes=NUM_CLASSES + 1, image_size=PATCH_SIZE)

# net = TransUnet.TransUnet(in_channels=PCA_NUM, img_dim=PATCH_SIZE, patch_size=5, classes=NUM_CLASSES + 1, type='seg')

# net = Differential_Transfofmer_UNet.DiTUnet(in_channels=PCA_NUM, img_dim=PATCH_SIZE, vit_blocks=8, vit_dim_linear_mhsa_block=32,classes=NUM_CLASSES + 1)

# net = SwinUnet.SwinUNet(H=224, W=224, ch=PCA_NUM, C=64, num_class=NUM_CLASSES + 1, num_blocks=3, patch_size=4)

# net = YNet.ResNetC1_YNet(H=PATCH_SIZE, W=PATCH_SIZE, in_channels=PCA_NUM, classes=NUM_CLASSES + 1, diagClasses= NUM_CLASSES + 1)

# net = Swin_Transformer_v2.SwinTransformerV2(in_chans=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = Swin_Transformer.SwinTransformer(in_chans=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = XNet.XNet(in_channels=PCA_NUM, num_classes=NUM_CLASSES + 1, x_in_channels=1)

# net = PNet.PNet2D(in_chns=PCA_NUM, out_chns=NUM_CLASSES + 1, num_filters=128, ratios=(2, 2, 2, 2, 2))

# net = VNet.VNet(n_channels=1, n_classes=NUM_CLASSES + 1)

# net = SegFormer.Segformer(channels=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = UNet3D.unet_3D(n_classes=NUM_CLASSES, in_channels=1)

# net = general_vit_2.General_ViT_2(channels=PCA_NUM, x_data_channel=1, num_classes=NUM_CLASSES + 1,
#                                   image_size=PATCH_SIZE, patch_size=5, type='seg')

# net = deepvit.DeepViT(image_size = PATCH_SIZE,num_classes = NUM_CLASSES + 1,channels=PCA_NUM)

# net = cait.CaiT(image_size=PATCH_SIZE, patch_size=3,num_classes=NUM_CLASSES+1,channels=PCA_NUM,)

# net = Differential_Transformer.DiT(channels=PCA_NUM, num_classes=NUM_CLASSES + 1, image_size=PATCH_SIZE, model_parallel_size=4)

# net = mobile_vit.MobileViT(image_size=(256, 256), num_classes=NUM_CLASSES+1, in_channels=PCA_NUM)

# net = pit.PiT(image_size=224, channels=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = cross_vit.CrossViT(image_size=224, num_classes=NUM_CLASSES + 1 , channels=PCA_NUM)

# net = TokenLearner.ViT(image_size=256, in_channels=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = regionvit.RegionViT(num_classes=NUM_CLASSES + 1, channels=PCA_NUM,)

# net = local_vit.LocalViT(image_size=PATCH_SIZE, patch_size=5, channels=PCA_NUM, num_classes=NUM_CLASSES + 1)

# net = RegionTransformer_v2.ViT(image_size=PATCH_SIZE, patch_size=5, num_classes=NUM_CLASSES + 1, channels=PCA_NUM, type='seg')

# net = HSIseg_v1.ViT_UNet(image_size=PATCH_SIZE, num_classes=NUM_CLASSES + 1, channels=PCA_NUM,x_data_channel=1)

# net = HSIseg_v2.ViT_UNet(image_size=PATCH_SIZE, num_classes=NUM_CLASSES + 1, channels=PCA_NUM, x_data_channel=1,)

# net = HSIseg_v3.ViT_UNet(image_size=PATCH_SIZE, num_classes=NUM_CLASSES + 1, channels=PCA_NUM, x_data_channel=1, )

# net = HSIseg_v4.ViT_UNet(image_size=PATCH_SIZE, num_classes=NUM_CLASSES + 1, channels=PCA_NUM,x_data_channel=1)
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def loadData(data_path: str = '../data', dtype: np.dtype = np.float64, run_times=20) -> tuple:
    """
    Load hyperspectral data, labels, and additional x_data from .mat files.

    Args:
        data_path: Path to the directory containing the .mat files.

    Returns:
        Tuple containing:
        - data: Normalized hyperspectral image data (H, W, C).
        - labels: Ground truth labels (H, W).
        - x_data: Normalized additional data (H, W, C or 1).
        - rgb: RGB visualization of the hyperspectral data (H, W, 3).
    """
    # Paths to the .mat files
    data_file = os.path.join(data_path, 'PaviaU.mat')
    x_data_file = os.path.join(data_path, 'simulated X_data/PU_simulated_LiDAR.mat')

    if run_times == 1:
        # First run: use the default ground truth file
        labels_file = os.path.join(data_path, 'PaviaU_gt.mat')
        print('Label file: Load original GT | PaviaU_gt.mat')
    elif 1 < run_times < RUN_TIMES:  #这里可以调整，取决于你是否需要用迭代后的模型来重新在原有GT上做新的训练去和第一次作比较
        # Subsequent runs: use the temporal updated file
        labels_file = os.path.join('temporal_gt', f'temporal_gt_updated_everytime_{run_times - 1}.mat')
        print(f'Label file: Load temporal GT | temporal_gt_updated_everytime_{run_times - 1}.mat')
    elif run_times == RUN_TIMES:  #这里可以调整，取决于你是否需要用迭代后的模型来重新在原有GT上做新的训练去和第一次作比较
        # Final run: potentially use the final GT or special handling (adjust based on your logic)
        labels_file = os.path.join(data_path, 'PaviaU_gt.mat')
        print('Label file: Load original GT for final run | PaviaU_gt.mat')
    else:
        raise ValueError("Invalid run_times value.")

    # Load data
    data = sio.loadmat(data_file)["paviaU"].astype(dtype)

    # labels_key = "paviaU_gt" if run_times == 1 else "temporal_gt"  # Handle different keys dynamically
    if run_times == 1 or run_times == RUN_TIMES:
        labels_key = "paviaU_gt"
    elif 1 < run_times < RUN_TIMES:  #这里可以调整，取决于你是否需要用迭代后的模型来重新在原有GT上做新的训练去和第一次作比较
        labels_key = "temporal_gt"
    else:
        raise ValueError(f"Invalid run_times: {run_times}. Expected 1 <= run_times <= {RUN_TIMES}.")

    labels = sio.loadmat(labels_file)[labels_key].astype(dtype)
    x_data = sio.loadmat(x_data_file)['lidar_image_1'].astype(dtype)

    # Normalize data and x_data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

    # Generate RGB visualization from specific bands
    rgb = data[:, :, (55, 35, 15)].astype(np.float32)

    # Ensure output directory exists
    output_dir = 'classification_maps'
    os.makedirs(output_dir, exist_ok=True)

    # Save RGB visualization
    plt.figure()
    plt.imshow(rgb)
    plt.savefig(os.path.join(output_dir, 'False_RGB.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # Visualize x_data
    if x_data.ndim == 2:
        plt.figure()
        plt.imshow(x_data, cmap='jet')
        # plt.colorbar()
        plt.savefig(os.path.join(output_dir, 'X_data_visualize.png'), dpi=600, bbox_inches='tight')
        plt.close()
    elif x_data.ndim == 3:
        for index in range(x_data.shape[2]):
            plt.figure()
            plt.imshow(x_data[:, :, index], cmap='jet')
            # plt.colorbar()
            plt.savefig(os.path.join(output_dir, f'X_data_visualize_{index}.png'), dpi=600, bbox_inches='tight')
            plt.close()

    # Ensure x_data is 3D
    if x_data.ndim == 2:
        x_data = np.expand_dims(x_data, axis=2)

    # Handle NaN values
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        print("Warning: NaN values found in the data. These will be set to 0.")
        data[nan_mask] = 0
        x_data[nan_mask] = 0
        labels[nan_mask] = 0

    return data, labels, x_data, rgb


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X: np.ndarray, numComponents: int, whiten: bool = True) -> np.ndarray:
    """
    Apply PCA to a 3D hyperspectral image.

    Args:
        X: Input 3D array (Height x Width x Channels).
        numComponents: Number of principal components to retain.
        whiten: Whether to apply whitening during PCA (default: True).

    Returns:
        Transformed 3D array with reduced components.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError("Input X must be a 3D numpy array (Height x Width x Channels).")

    if not isinstance(numComponents, int) or numComponents <= 0 or numComponents > X.shape[2]:
        raise ValueError(f"numComponents must be an integer between 1 and {X.shape[2]}.")

    print('======>>> Starting PCA ======>>>')

    with tqdm(total=3, desc='PCA Progress') as pbar:
        try:
            # Step 1: Reshape the input
            newX = np.reshape(X, (-1, X.shape[2]))
            pbar.update(1)  # Update progress

            # Step 2: Apply PCA
            pca = PCA(n_components=numComponents, whiten=whiten)
            newX = pca.fit_transform(newX)
            pbar.update(1)  # Update progress

            # Step 3: Reshape the output
            newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
            pbar.update(1)  # Update progress
        except Exception as e:
            print(f"Error during PCA: {e}")
            raise

    rgb = newX[:, :, 0:3]
    plt.figure()  # Create a new figure
    plt.imshow(rgb)
    plt.savefig('classification_maps/False_RGB_with_3_PCA.png', dpi=600, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # for i in tqdm(range(numComponents), desc='Store PCA components in folder:'):
    #     PC = newX[:, :, i]
    #     plt.figure()  # Create a new figure
    #     plt.imshow(PC, cmap='grey')
    #     plt.colorbar()
    #     plt.savefig(f'PCA_component/PC_{i}.png', dpi=600, bbox_inches='tight')
    #     plt.close()  # Close the figure to free memory
    # print('======>>> Finishing PCA =====>>>')
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X: np.ndarray, margin: int = int(PATCH_SIZE // 2)) -> np.ndarray:
    """
    Pad a 2D or 3D input (e.g., hyperspectral or auxiliary data) with zeros symmetrically.

    Args:
        X: Input array, 2D (height, width) or 3D (height, width, channels).
        margin: Size of the zero-padding on each side (default: 2).

    Returns:
        np.ndarray: Padded array with the same dtype as the input.
    """
    if margin < 0:
        raise ValueError("Margin must be non-negative.")
    if X.size == 0:
        raise ValueError("Input array must not be empty.")

    # Determine the shape of the padded array
    if X.ndim == 3:
        padded_shape = (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2])
        newX = np.zeros(padded_shape, dtype=X.dtype)
        # Insert original data into the center
        newX[margin:-margin, margin:-margin, :] = X
    elif X.ndim == 2:
        padded_shape = (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin)
        newX = np.zeros(padded_shape, dtype=X.dtype)
        # Insert original data into the center
        newX[margin:-margin, margin:-margin] = X
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return newX


class get_dataset(torch.utils.data.Dataset):
    """Dataset class for a hyperspectral scene with auxiliary data."""

    def __init__(self, data, x_data, gt, patch_size=PATCH_SIZE, ignored_labels=[0], supervision='full'):
        """
        Args:
            data: 3D hyperspectral image (H x W x C).
            x_data: Auxiliary data (e.g., another image modality) with the same spatial dimensions as `data`.
            gt: 2D array of ground truth labels (H x W).
            patch_size: Size of the patch window.
            ignored_labels: List of labels to ignore during training.
            supervision: 'full' or 'semi', determines how to handle ignored labels.
        """
        super(get_dataset, self).__init__()
        self.data = padWithZeros(data, margin=int(patch_size // 2))
        self.x_data = padWithZeros(x_data, margin=int(patch_size // 2))
        self.gt = padWithZeros(gt, margin=int(patch_size // 2))
        self.x_data = padWithZeros(x_data)
        self.gt = padWithZeros(gt)
        self.patch_size = patch_size
        self.ignored_labels = ignored_labels
        self.supervision = supervision

        # Mask valid indices
        self.indices = self._create_valid_indices()

        # Shuffle indices for randomness
        np.random.shuffle(self.indices)

        # Preload labels for each patch center
        self.labels = np.array([self.gt[x, y] for x, y in self.indices])

    def _create_valid_indices(self):
        """Create valid indices for patches, excluding borders and ignored labels."""
        mask = np.ones_like(self.gt, dtype=bool)

        if self.supervision == "full":
            for label in self.ignored_labels:
                mask[self.gt == label] = False
        elif self.supervision != "semi":
            raise ValueError(f"Unsupported supervision type: {self.supervision}")

        p = self.patch_size // 2
        x_pos, y_pos = np.nonzero(mask)
        return np.array([
            (x, y) for x, y in zip(x_pos, y_pos)
            if p <= x < self.data.shape[0] - p and p <= y < self.data.shape[1] - p
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a patch, its auxiliary data, and the corresponding labels.

        Args:
            idx: Index of the sample.

        Returns:
            data: Tensor containing the hyperspectral patch (C x H x W).
            x_data: Tensor containing the auxiliary data patch (C x H x W).
            patch_labels: Tensor of the labels for the entire patch.
            patch_center_label: Tensor of the center pixel's label.
        """
        x, y = self.indices[idx]
        p = self.patch_size // 2
        x1, y1, x2, y2 = x - p, y - p, x + p + 1, y + p + 1

        # Extract patches
        data_patch = self.data[x1:x2, y1:y2]
        x_data_patch = self.x_data[x1:x2, y1:y2]
        label_patch = self.gt[x1:x2, y1:y2]

        # Convert patches to tensors
        data_patch = torch.tensor(data_patch.transpose((2, 0, 1)), dtype=torch.float32)
        data_patch = data_patch.unsqueeze(0)
        if x_data_patch.ndim == 3:
            x_data_patch = torch.tensor(x_data_patch.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        else:
            x_data_patch = torch.tensor(x_data_patch.transpose(0, 1), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        patch_labels = torch.tensor(label_patch, dtype=torch.int64)
        patch_center_label = torch.tensor(self.gt[x, y], dtype=torch.int64) - 1  #减少1是为了让模型忽略掉background
        # print(
        # f'max value in patch_labels: {patch_labels.max().item()} | min value in patch_labels: {patch_labels.min().item()} | max value in patch_center_label: {patch_center_label.max().item()} | min value in patch_center_label: {patch_center_label.min().item()}')
        return data_patch, x_data_patch, patch_labels, patch_center_label


def sample_gt(gt, train_size, mode='random'):
    """
    Extract a fixed number of samples or a percentage of samples from an array of labels for training and testing.

    Args:
        gt: a 2D array of int labels
        train_size: an int number of samples or a float percentage [0, 1] of samples to use for training
        mode: a string specifying the sampling strategy, options include 'random', 'fixed', 'disjoint'

    Returns:
        train_gt: a 2D array of int labels for training
        test_gt: a 2D array of int labels for testing
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if isinstance(train_size, float):
        train_size = int(train_size * y.size)

    if mode == 'random':
        train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y if train_size > 1 else None)
        train_gt[tuple(zip(*train_indices))] = gt[tuple(zip(*train_indices))]
        test_gt[tuple(zip(*test_indices))] = gt[tuple(zip(*test_indices))]
    elif mode == 'fixed':
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = np.array(list(zip(*indices)))  # x,y features
            total_samples = len(X)

            # Ensure at least one sample is left for testing
            adjusted_train_size = min(train_size, total_samples - 1) if total_samples > 1 else 1

            train_indices, test_indices = train_test_split(X, train_size=adjusted_train_size, stratify=None)

            train_gt[tuple(zip(*train_indices))] = gt[tuple(zip(*train_indices))]
            test_gt[tuple(zip(*test_indices))] = gt[tuple(zip(*test_indices))]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError(f"{mode} sampling is not implemented yet.")

    return train_gt, test_gt


def create_data_loader(data_path="../data", batch_size=BATCH_SIZE_TRAIN, sample_type=USE_SAMPLE_TYPE, run=RUN_TIMES):
    """
    Prepare data loaders for training and testing.

    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.
        train_sample_ratio (float): Ratio of training samples (used if sample_type is 'random_ratio').
        fixed_number_per_class (int): Fixed number of samples per class (used if sample_type is 'fix_number').
        sample_type (str): Sampling type ('random_ratio', 'fix_number', or 'exist').
        num_workers (int): Number of workers for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]: Train loader, test loader, raw X, and raw y.
    """
    # Load data
    X, y, x_data, rgb = loadData(data_path, dtype=np.float64, run_times=run)
    print(f'\nHyperspectral data shape: {X.shape}')
    if PCA_NUM != X.shape[2]:
        X_pca = applyPCA(X, numComponents=PCA_NUM)
    print(
        f'Hyperspectral data shape after PCA: {X_pca.shape}\nAuxiliary data shape: {x_data.shape}\nLabel shape: {y.shape}')

    print('\n... Splitting data into train & test ...')
    if sample_type == "random":
        print(f'RANDOM_SAMPLEING: TRAINING RATIO: {TRAIN_SAMPLE_RATIO}')
        train_gt, test_gt = sample_gt(y, train_size=TRAIN_SAMPLE_RATIO, mode='random')
    elif sample_type == "fix":
        print(f"FIXED SAMPLING: TRAINING NUMBER PER CLASS: {FIXED_NUMBER_PER_CLASS}")
        train_gt, test_gt = sample_gt(y, train_size=FIXED_NUMBER_PER_CLASS, mode='fixed')
    elif sample_type == "exist":
        print(f'EXISTED_SAMPLES_SAMPLEING MODE')  #MUUFL数据集没有已存的训练集和测试集，只能用fix或者random的采样
        train_gt = sio.loadmat(f'../data/PaviaU_index.mat')['TR']
        test_gt = sio.loadmat(f'../data/PaviaU_index.mat')['TE']
    else:
        raise ValueError(f"Invalid sample_type: {sample_type}. Choose from 'random_ratio', 'fix_number', or 'exist'.")
    print("{} samples selected (over {}) | Ratio {}".format(np.count_nonzero(train_gt), np.count_nonzero(y),
                                                            np.count_nonzero(train_gt) / np.count_nonzero(y)))

    print("{} samples selected (over {}) | Ratio {}".format(np.count_nonzero(test_gt), np.count_nonzero(y),
                                                            np.count_nonzero(test_gt) / np.count_nonzero(y)))
    print(f"Number of classes: {len(np.unique(train_gt)) - 1}")
    print(f"Number of classes: {len(np.unique(test_gt)) - 1}")

    # Convert train_gt and test_gt to color
    train_gt_display = get_cls_map_v3.convert_to_color_(train_gt, palette=get_cls_map_v3.palette)
    test_gt_display = get_cls_map_v3.convert_to_color_(test_gt, palette=get_cls_map_v3.palette)
    get_cls_map_v3.classification_map(train_gt_display, 600, f'classification_maps/train_gt_{run}.png')
    get_cls_map_v3.classification_map(test_gt_display, 600, f'classification_maps/test_gt_{run}.png')

    # Create datasets and DataLoaders
    train_dataset = get_dataset(X_pca, x_data, train_gt)
    print(f"Total training samples: {len(train_dataset)}")
    test_dataset = get_dataset(X_pca, x_data, test_gt)
    print(f"Total test samples: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, train_gt, test_loader, test_gt, X_pca, y, x_data, rgb


def train(net, train_loader, epochs, run):
    summary(net, input_size=[(BATCH_SIZE_TRAIN, 1, PCA_NUM, PATCH_SIZE, PATCH_SIZE),
                             (BATCH_SIZE_TRAIN, 1, 1, PATCH_SIZE, PATCH_SIZE)])

    # Define loss functions
    criterion_out_seg = nn.CrossEntropyLoss(ignore_index=0)
    criterion_out_cls = nn.CrossEntropyLoss()
    criterion_aux_seg = nn.CrossEntropyLoss(ignore_index=0)
    criterion_aux_rec = nn.MSELoss()
    dice = MultiClassDiceLoss(ignore_index=None)
    center_criterion = CenterLoss(num_classes=NUM_CLASSES, feature_dim=NUM_CLASSES, device=device)

    # Initialize trainable parameters
    alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))
    beta = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))
    gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))
    theta = nn.Parameter(torch.tensor(1.0, requires_grad=True, device=device))

    # Add parameters to optimizer
    optimizer = optim.AdamW([
        {'params': net.parameters()},
        {'params': [alpha, beta, gamma, theta]},
    ], lr=0.001)

    # Initialize lists to track weights
    alpha_values, beta_values, gamma_values, theta_values = [], [], [], []

    for epoch in tqdm(range(epochs), desc="Training the network", colour='red'):
        net.train()
        total_loss = 0

        # Calculate softmax weights
        softmax_weights = F.softmax(torch.stack([alpha, beta, gamma, theta]), dim=0)
        a, b, c, d = softmax_weights

        # Store the weights for this epoch
        alpha_values.append(a.item())
        beta_values.append(b.item())
        gamma_values.append(c.item())
        theta_values.append(d.item())

        for data, aux_data, target_patch, target in tqdm(train_loader,
                                                         desc="Loading Train Loaders in One Epoch", colour='blue'):
            data, aux_data, target_patch, target = (
                data.to(device), aux_data.to(device), target_patch.to(device), target.to(device)
            )

            # Forward pass
            outputs = net(data, aux_data)
            loss = 0

            if len(outputs) == 4:
                out_seg, out_cls, aux_seg, aux_rec = outputs
                out_cls = out_cls[:, 1:]
                loss += a * criterion_out_seg(out_seg, target_patch)
                loss += d * (
                        criterion_out_cls(out_cls, target)
                        + center_criterion(out_cls, target)
                        + contrastive_loss(out_cls, target)
                )
                loss += b * dice(aux_seg, target_patch)
                loss += c * criterion_aux_rec(aux_rec, data.squeeze(1))
            elif len(outputs) == 3:
                out_seg, out_cls, rec = outputs
                out_cls = out_cls[:, 1:]
                loss += a * criterion_out_seg(out_seg, target_patch)
                loss += b * (
                        criterion_out_cls(out_cls, target)
                        + center_criterion(out_cls, target)
                        + contrastive_loss(out_cls, target)
                )
                loss += c * criterion_out_seg(rec, target_patch)
            elif len(outputs) == 2:
                out_seg, out_cls = outputs
                out_cls = out_cls[:, 1:]
                loss += criterion_out_seg(out_seg, target_patch)
                loss += b * (
                        criterion_out_cls(out_cls, target)
                        + center_criterion(out_cls, target)
                        + contrastive_loss(out_cls, target)
                )
            else:
                if outputs.ndim == 4:
                    out_seg = outputs
                    loss += criterion_out_seg(out_seg, target_patch)
                elif outputs.ndim == 2:
                    out_cls = outputs
                    out_cls = out_cls[:, 1:]
                    loss += (
                            criterion_out_cls(out_cls, target)
                            + center_criterion(out_cls, target)
                            + contrastive_loss(out_cls, target))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Track total loss
            total_loss += loss.item()

        # Calculate average loss per epoch
        average_loss = total_loss / len(train_loader)

        # Print epoch statistics
        print(f"\nRun {run}/{RUN_TIMES}")
        print(f"Epoch: {epoch + 1} | Average Loss: {average_loss:.6f}")
        print(
            f"Weights for Loss Item -> a: {a.item():.4f}, b: {b.item():.4f}, c: {c.item():.4f}, "
            f"d: {d.item():.4f}"
        )

    # Plot weight evolution
    plot_weights(alpha_values, beta_values, gamma_values, theta_values, epochs)
    return net, device


def plot_weights(alpha_values, beta_values, gamma_values, theta_values, epochs):
    """
    Plot the evolution of weights across epochs.
    """
    plt.figure()
    plt.plot(range(1, epochs + 1), alpha_values, label='Alpha', marker='o')
    plt.plot(range(1, epochs + 1), beta_values, label='Beta', marker='s')
    plt.plot(range(1, epochs + 1), gamma_values, label='Gamma', marker='^')
    plt.plot(range(1, epochs + 1), theta_values, label='Theta', marker='x')

    plt.title('Evolution of Loss Weights Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('weight_evolution.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def get_test_features(device, net, test_loader):
    net.eval()
    features = []
    labels = []

    with torch.inference_mode():
        for data, x_data, target_patch, target in tqdm(test_loader, desc='Prepaing features for TSNE on test samples'):
            # Move data to device
            data = data.to(device)
            x_data = x_data.to(device)
            target = target.to(device)

            # Forward pass
            outputs = net(data, x_data)
            if len(outputs) == 4:
                out_seg, out_cls, aux_seg, aux_rec = outputs
                out_cls = out_cls[:, 1:]
            elif len(outputs) == 3:
                out_seg, out_cls, aux_rec = outputs
                out_cls = out_cls[:, 1:]
            elif len(outputs) == 2:
                out_seg, out_cls = outputs
                out_cls = out_cls[:, 1:]
            else:
                out_cls = outputs
                out_cls = out_cls[:, 1:]

            # Get predictions and ground truth
            ground_truth = target.detach().cpu().numpy()
            features.append(out_cls.detach().cpu().numpy())
            labels.append(ground_truth)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def plot_tsne(features, labels, n_components=2, perplexity=30, save_path='tsne_plot.png'):
    print(f'\nNumber of test samples: {len(features)} | Number of Classes: {NUM_CLASSES} | Min label: {min(labels)} | '
          f'Max label: {max(labels)}')
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    with tqdm(total=len(features), desc='Running t-SNE on test samples (Need wait)') as pbar:
        tsne_results = tsne.fit_transform(features)
        pbar.update(len(features))

    plt.figure()
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='jet', edgecolors='white')
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title('t-SNE Visualization of Test Features')
    plt.grid()
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'\nt-SNE plot saved to {save_path}')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)  # Ensure the model is moved to the correct device (GPU or CPU)
    threshold = PROBABILITIC_MASK
    for run in range(RUN_TIMES):
        print(f"\n\nRun {run + 1}/{RUN_TIMES}")

        ## Load data
        train_loader, train_gt, test_loader, test_gt, X_pca, y, x_data, rgb = create_data_loader(run=run + 1)

        # Timing training
        tic1 = time.perf_counter()
        net, device = train(net.to(device), train_loader, epochs=EPOCH, run=run + 1)
        toc1 = time.perf_counter()
        training_time = toc1 - tic1

        # Timing testing
        tic2 = time.perf_counter()
        features, classes = get_test_features(device, net, test_loader)
        toc2 = time.perf_counter()
        test_time = toc2 - tic2

        # Plot t-SNE visualization on test samples
        plot_tsne(features, classes, save_path=f'TSNE_results/tsne_plot_{run + 1}.png')

        # 只保存模型参数
        torch.save(net.state_dict(), f'cls_params/params_{run + 1}.pth')

        # Generate classification maps (or temporal GT)
        get_cls_map_v3.get_cls_map(net=net, img=X_pca, x_data=x_data, gt=y, test_gt=test_gt, rgb=rgb, run=run + 1,
                                   threshold=threshold)
        threshold = threshold - TOLERANCE  # 随次数改变的MASK阈值,你可以越来越严格或者越来越松弛也可以保持不变

    print("All runs completed and saved.")
