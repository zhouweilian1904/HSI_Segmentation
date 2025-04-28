import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tqdm
import torch
from PU_train_v3 import PATCH_SIZE, NUM_CLASSES, BATCH_SIZE_TRAIN, RUN_TIMES, PROBABILITIC_MASK, TOLERANCE
import torch.nn.functional as F
import itertools
import os
from scipy.io import savemat
import scipy.io as sio
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score

target_names = [
        "1.Asphalt",
        "2.Meadows",
        "3.Gravel",
        "4.Trees",
        "5.Metal sheets",
        "6.Bare Soil",
        "7.Bitumen",
        "8.Bricks",
        "9.Shadows",
    ]

palette = {
    0: np.array([0, 0, 0]),  # Black
    1: np.array([255, 0, 0]),  # Red
    2: np.array([0, 255, 0]),  # Green
    3: np.array([0, 0, 255]),  # Blue
    4: np.array([255, 255, 0]),  # Yellow
    5: np.array([255, 0, 255]),  # Magenta
    6: np.array([0, 255, 255]),  # Cyan
    7: np.array([128, 0, 0]),  # Dark Red
    8: np.array([0, 128, 0]),  # Dark Green
    9: np.array([0, 0, 128]),  # Dark Blue
    10: np.array([128, 128, 0]),  # Olive
    11: np.array([128, 0, 128]),  # Purple
    12: np.array([0, 128, 128]),  # Teal
    13: np.array([192, 192, 192]),  # Light Grey
    14: np.array([64, 64, 64]),  # Dark Grey
    15: np.array([255, 128, 0]),  # Orange
    16: np.array([128, 128, 255]),  # Light Blue
    17: np.array([255, 192, 203]),  # pink
    18: np.array([165, 42, 42]), # brown
    19: np.array([255, 215, 0]),  # gold
    20: np.array([75, 0, 130]), # indigo
}


def count_sliding_window(image, step=10, window_size=(20, 20)):
    """Count the number of windows in an image."""
    sw = sliding_window(image, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """Group n elements at a time from the iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image."""
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step

    for x in range(0, W - w + offset_w + 1, step):
        for y in range(0, H - h + offset_h + 1, step):
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def test_on_all(net, img, x_data, patch_size, batch_size, n_classes, run_times, threshold):
    """
    Test a model on a specific image, with an additional x_data input.
    """
    print(f'img shape: {img.shape} | x_data shape: {x_data.shape}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()

    # Ensure inputs match model parameter dtype
    model_dtype = next(net.parameters()).dtype

    kwargs = {
        "step": 1,
        "window_size": (patch_size, patch_size),
    }

    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
            grouper(batch_size, zip(sliding_window(img, **kwargs), sliding_window(x_data, **kwargs))),
            total=iterations,
            desc="Generating Classification Map",
            colour='blue'
    ):
        with torch.inference_mode():
            indices = [b[0][1:] for b in batch]

            # Convert img data to tensor
            data = np.stack([b[0][0] for b in batch])  # Directly stack
            data = data.transpose(0, 3, 1, 2)  # (B, C, H, W)
            data = torch.from_numpy(data).to(device=device, dtype=model_dtype)
            data = data.unsqueeze(1)

            # Convert x_data to tensor
            x_data = np.stack([b[1][0] for b in batch])  # Directly stack
            x_data = x_data.transpose(0, 3, 1, 2)  # (B, C, H, W)
            x_data = torch.from_numpy(x_data).to(device=device, dtype=model_dtype)
            x_data = x_data.unsqueeze(1)

            # Pass both data and x_data to the model
            output = net(data, x_data)
            if isinstance(output, tuple):
                output = output[0]
                output = F.softmax(output, dim=1)

            output = output.cpu().numpy()

            if output.ndim == 2:  #out_cls
                output = output
            else:
                output = np.transpose(output, (0, 2, 3, 1))  #out_seg ndim==4

            for (x, y, w, h), out in zip(indices, output):
                if output.ndim == 2:  #out_cls 如果想用单个输出，建议用另外两个版本的train或者train_v2
                    probs[x + w // 2, y + h // 2] += out
                else:  #out_seg ndim==4
                    probs[x: x + w, y: y + h] += out

    # Normalize to ensure probabilities sum to 1 per pixel
    probs /= np.sum(probs, axis=-1, keepdims=True)  # Normalize probabilities

    probs_without_mask = probs.copy()
    # Find the maximum probability for each pixel
    max_probs = np.max(probs, axis=-1)  # Shape: (height, width)
    mean_probs = np.mean(max_probs)
    std_probs = np.std(max_probs)

    # Adaptive threshold based on run_times
    adapt_threshold = mean_probs + std_probs
    if adapt_threshold > threshold:
        adapt_threshold = threshold
    print(f'adapt_threshold: {adapt_threshold}')
    if run_times < RUN_TIMES:  # 如果RUN_TIMES等于1，则表示supervised learning，如果大于1，则是semi-supervised learning (progressive pesudo labelling)
        # Apply the mask: set all probabilities to 0 for masked pixels
        mask = max_probs < adapt_threshold  # Mask pixels where max probability < 0.8
        probs[mask] = 0  # This will zero out all class probabilities for masked pixels
        plot_probs(probs, n_classes, run=run_times)
    else:
        probs = probs
        plot_probs(probs, n_classes, run=run_times)
    return probs, probs_without_mask


def plot_probs(probs, n_classes, run):
    """
    Visualize the probabilities from the `probs` array.

    Args:
        probs (numpy.ndarray): Array of probabilities with shape (height, width, n_classes).
        n_classes (int): Total number of classes.

    Returns:
        None. Displays the boxplot.
    """
    # Get predicted class for each pixel
    predicted_classes = np.argmax(probs, axis=-1)

    # Prepare data for boxplot
    boxplot_data = []
    pixel_counts = []

    for cls in range(n_classes):
        # Mask for pixels predicted as this class
        class_mask = predicted_classes == cls

        # Extract probabilities for this class
        class_probs = probs[class_mask, cls]

        # Append to boxplot data
        boxplot_data.append(class_probs)

        # Count number of pixels predicted for this class
        pixel_counts.append(len(class_probs))

    # Plot boxplot
    plt.figure()
    plt.boxplot(boxplot_data, labels=[f'{i}' for i in range(n_classes)])
    plt.xlabel("Classes")
    plt.ylabel("Probabilities")
    plt.title("Distribution of Probabilities for Each Predicted Class")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'temporal_gt/Distribution of Probabilities_{run}.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Plot pixel counts per class
    plt.figure()
    plt.bar(range(n_classes), pixel_counts, tick_label=[f'{i}' for i in range(n_classes)])
    plt.xlabel("Classes")
    plt.ylabel("Pixel Count")
    plt.title("Pixel Counts for Each Predicted Class")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'temporal_gt/Pixel Counts_{run}.png', dpi=600, bbox_inches='tight')
    plt.close()


def convert_to_color_(arr_2d: np.ndarray, palette: dict = None) -> np.ndarray:
    """Convert an array of labels to RGB color-encoded image."""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    if palette is None:
        # Generate a default palette with random colors
        palette = {0: np.array([0, 0, 0])}
        for k in range(1, arr_2d.max() + 1):
            palette[k] = np.random.randint(0, 256, size=3)

    for c, color in palette.items():
        # Mask for the current label
        mask = (arr_2d == c)
        arr_3d[mask] = color.astype(np.uint8)  # Assign color

    return arr_3d


def classification_map(map, dpi, save_path):
    plt.figure()
    plt.imshow(map)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return 0


def metrics(prediction, target, ignored_labels=[0], n_classes=NUM_CLASSES + 1):
    """Compute and print metrics (accuracy, confusion matrix, F1 scores, etc.).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g., 0 for undefined
        n_classes (optional): number of classes, max(target) by default

    Returns:
        Dictionary of results including confusion matrix, accuracies, F1 scores, and kappa.
    """
    # Mask ignored labels
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    # Flatten arrays (ensure they are 1D sequences)
    target = target.flatten()
    prediction = prediction.flatten()

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    # Confusion matrix
    cm = confusion_matrix(target, prediction, labels=range(n_classes))
    results["Confusion matrix"] = cm

    # Global Accuracy
    total = np.sum(cm)
    overall_accuracy = np.trace(cm) / float(total)
    results["Overall Accuracy"] = overall_accuracy * 100

    # Class Accuracy
    class_accuracies = []
    for i in range(len(cm)):
        if i == 0:  # Skip ignored label
            continue
        total_samples = np.sum(cm[i, :])
        if total_samples > 0:
            class_accuracy = cm[i, i] / total_samples
        else:
            class_accuracy = 0  # No samples for this class
        class_accuracies.append(class_accuracy * 100)
    results["Class Accuracies"] = class_accuracies

    # Average Accuracy
    average_accuracy = np.mean(class_accuracies) if len(class_accuracies) > 0 else 0
    results["Average Accuracy"] = average_accuracy

    # Cohen's Kappa
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def show_results(results, label_values=target_names, run=RUN_TIMES):
    print("\nShowing Results | Comparing the alignment with test_gt (not just using test_loader before) |")
    text = ""

    cm = results["Confusion matrix"]
    overall_accuracy = results["Overall Accuracy"]
    average_accuracy = results["Average Accuracy"]
    class_accuracies = results["Class Accuracies"]
    kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "\n---\n"

    text += "Overall Accuracy : {:.06f}%\n".format(overall_accuracy)
    text += "Average Accuracy : {:.06f}%\n".format(average_accuracy)
    text += "---\n"

    text += "Class Accuracies :\n"
    for label, class_accuracy in zip(label_values, class_accuracies):
        text += "\t{}: {:.06f}%\n".format(label, class_accuracy)
    text += "---\n"

    text += "Kappa: {:.06f}\n".format(kappa)

    # Save individual run results to file
    file_name = f"cls_result/comparing_with_test_gt_run_{run}.txt"
    with open(file_name, 'w') as x_file:
        x_file.write(f"Run {run}\n\n")
        x_file.write(f"Overall accuracy (%): {overall_accuracy}\n")
        x_file.write(f"Average accuracy (%): {average_accuracy}\n")
        x_file.write(f"Each accuracy (%): {class_accuracies}\n")
        x_file.write(f"Kappa accuracy (%): {kappa}\n\n")
        x_file.write(f"Confusion Matrix:\n{str(cm)}\n\n")
    print(text)


def get_cls_map(net, img, x_data, gt, test_gt, rgb, run, threshold, ):
    # Generate probabilities (H, W, num_classes)
    probabilities, probabilities_without_mask = test_on_all(net=net, img=img, x_data=x_data, patch_size=PATCH_SIZE,
                                                            batch_size=BATCH_SIZE_TRAIN,
                                                            n_classes=NUM_CLASSES + 1, run_times=run,
                                                            threshold=threshold, )

    # Generate predictions with threshold
    prediction1 = np.argmax(probabilities, axis=-1)  # (H, W)
    prediction2 = prediction1.copy()  # Create a copy for modification
    # Generate predictions without threshold (intermediate maps)
    prediction3 = np.argmax(probabilities_without_mask, axis=-1)

    # Create masks
    # mask1 = (y == 0)  # Mask where ground truth is 0
    # # Load the .mat file
    mat_file = os.path.join('../data', 'PaviaU_gt.mat')
    mat_data = sio.loadmat(mat_file)  # Load the file as a dictionary
    # Access the "paviaU_gt" key
    original_y = mat_data["paviaU_gt"]
    mask1 = (original_y == 0)  # Mask where ground truth is 0

    # same_index_area_mask = prediction1 != original_y # constrain the sampling area where they share the same

    # Apply masks
    prediction1[mask1] = 0  # Masked prediction

    # Convert predictions to color
    color_prediction1 = convert_to_color_(prediction1, palette=palette)
    color_prediction2 = convert_to_color_(prediction2, palette=palette)
    color_prediction3 = convert_to_color_(prediction3, palette=palette)
    color_ground_truth = convert_to_color_(gt, palette=palette)

    # Ensure output directories exist
    os.makedirs('classification_maps', exist_ok=True)
    os.makedirs('position_map', exist_ok=True)

    # Save classification maps
    classification_map(color_ground_truth, 600, 'classification_maps/original_gt_or_temporal_gt.png')
    classification_map(color_prediction1, 600, f'classification_maps/predictions_masked_{run}.png')
    classification_map(color_prediction2, 600, f'classification_maps/predictions_no_masked_{run}.png')
    classification_map(color_prediction2, 600, f'temporal_gt/temporal_gt_updated_everytime_{run}.png')
    classification_map(color_prediction3, 600, f'temporal_gt/intermediate_result_{run}.png')

    # Save the array into the specified file
    savemat(f'temporal_gt/temporal_gt_updated_everytime_{run}.mat', {'temporal_gt': prediction2})
    print(f"File saved successfully as 'temporal_gt/temporal_gt_updated_everytime_{run}.mat'")

    # 另外一种方法来计算测试集上的精度
    run_results = metrics(prediction3, test_gt, ignored_labels=[0])
    show_results(run_results, label_values=target_names, run=run)

    # Save class position map
    # get_class_position(classification_map=prediction2, rgb=rgb, save_path='position_map/', current_time=current_time)

    print('------Get classification maps successful-------')


def get_class_position(classification_map: np.ndarray, rgb: np.ndarray, save_path: str, current_time: str):
    """
    Generate and save position maps for each class in the classification map.

    Args:
        classification_map: 2D array of class labels.
        rgb: RGB image for overlay.
        save_path: Directory to save the position maps.
        current_time: Timestamp string for file naming.
    """
    if classification_map.shape[:2] != rgb.shape[:2]:
        raise ValueError("Classification map and RGB image must have the same height and width.")
    print(f'classification_map shape: {classification_map.shape} | rgb shape: {rgb.shape}')

    # Round and cast the classification map to integers
    classification_map = np.round(classification_map).astype(int)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Iterate over each unique class label
    unique_labels = np.unique(classification_map)
    for class_label in tqdm(unique_labels, desc="Processing class_positions"):
        if class_label == 0:
            # Skip background or unlabeled areas
            continue

        # Create a mask for the current class
        mask = (classification_map == class_label)

        # Create an overlay for the current class
        overlay = np.copy(rgb)
        color = palette.get(class_label, np.array([255, 255, 255]))  # Default to white if label missing
        overlay[mask] = color.astype(np.uint8)

        # Save the overlay image
        save_filename = os.path.join(save_path, f'class_{class_label}_position_{current_time}.png')
        plt.figure()
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
        plt.close()
