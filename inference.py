import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from numba import jit
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

warnings.filterwarnings("ignore")

from get_data_dfs import load_config, process_data
from get_data_loaders import get_data_loaders

# Load config and process data
config = load_config()


train_df, valid_df, test_df = process_data(config)
test_loader = get_data_loaders(
    inference=True,
    test_df=test_df,
    batch_size=1,
    num_workers=0,
)


def list_jpeg_images(directory):
    """Returns a list of paths to JPEG images in the specified directory."""
    return [str(p) for p in Path(directory).rglob("*.jpg")]


class_to_int = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}

int_to_class = {v: k for k, v in class_to_int.items()}

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

dir_train = os.path.join(parent_dir, config["dir_train"])
dir_test = os.path.join(parent_dir, config["dir_test"])
out_dir = os.path.join(parent_dir, config["out_dir"])

for subdir in ["0", "1", "2", "3"]:
    subdir_path = os.path.join(out_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

# Set the target image size
size = 512

# Define the file with all the labels in the COCO-format
file = os.path.join(os.getcwd(), "result.json")
#
# # Inputs required for training
# model_path = os.path.join(os.getcwd(), "fasterrcnn_resnet50_fpn2_17July.pth")
# early_stop = 1  # This is required for early stopping, the number of epochs we will wait with no improvement before stopping
#
#
# # Load a pretrained model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#     box_detections_per_img=1, pretrained=True
# )
#
#
# num_classes = 2  # 1 class (OCT) + background
#
# # Get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# # Replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# Function to calculate loss for every epoch
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


@jit(nopython=True)
def calculate_iou(gt, pr, form="pascal_voc") -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == "coco":
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
        (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
        + (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1)
        - overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(
    gts, pred, pred_idx, threshold=0.5, form="pascal_voc", ious=None
) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(gts, preds, threshold=0.5, form="coco", ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(
            gts, preds[pred_idx], pred_idx, threshold=threshold, form=form, ious=ious
        )

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds=(0.5,), form="coco") -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(
            gts.copy(), preds, threshold=threshold, form=form, ious=ious
        )
        image_precision += precision_at_threshold / n_threshold

    return image_precision


from ensemble_boxes import *


def make_ensemble_predictions(images):
    images = list(image.to(device) for image in images)
    result = []

    model.eval()
    outputs = model(images)
    result.append(outputs)

    return result


def run_wbf(
    predictions,
    image_index,
    image_size=640,
    iou_thr=0.55,
    skip_box_thr=0.5,
    weights=None,
):
    boxes = [
        prediction[image_index]["boxes"].data.cpu().numpy() / (image_size - 1)
        for prediction in predictions
    ]
    scores = [
        prediction[image_index]["scores"].data.cpu().numpy()
        for prediction in predictions
    ]
    labels = [
        np.ones(prediction[image_index]["scores"].shape[0])
        for prediction in predictions
    ]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


# Inputs required for training
model_path = os.path.join(os.getcwd(), "fasterrcnn_resnet50_fpn2_17July.pth")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Current device:", device)


model = torch.load(model_path)


test_df = pd.DataFrame()
images_path_test = list_jpeg_images(dir_train)
test_df["image_id"] = images_path_test


model.eval()


# Create an empty DataFrame
result_df = pd.DataFrame(
    columns=["path", "shapex", "shapey", "x", "y", "width", "height"]
)


# Create a function that adjusts brightness
def adjust_brightness(image):
    # Find the minimum and maximum pixel values
    min_value = np.min(image)
    max_value = np.max(image)

    # Calculate the scaling factor
    scaling_factor = max_value - min_value

    # Adjust the brightness of the image
    adjusted_image = image - min_value
    normalized_image = adjusted_image / scaling_factor

    return normalized_image


# Cre
torch.cuda.empty_cache()

detection_threshold = 0.5
results = []
outputs = []
resize_transform = transforms.Resize((size, size))

for images, image_ids in tqdm(test_loader):
    images = list(image.to(device) for image in images)
    predictions = make_ensemble_predictions(images)

    for i, image in enumerate(images):
        w = image.shape[2]
        h = image.shape[1]
        s = max(w, h)
        scale = max(w, h) / min(w, h)

        # Resize the image to s * s
        image = adjust_brightness(image.permute(1, 2, 0).cpu().numpy())
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = cv2.resize(image.permute(1, 2, 0).cpu().numpy(), (s, s))
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes = boxes.astype(np.int32).clip(min=0, max=s - 1)

        preds = boxes
        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = preds[preds_sorted_idx]
        boxes = preds

        # Crop the image using the bounding box coordinates
        try:
            topmost_y = min(boxes[:, 1])
        except:
            topmost_y = 0

        try:
            bottommost_y = max(boxes[:, 3])
        except:
            bottommost_y = s - 1

        bottommost_y = int(bottommost_y * scale)
        topmost_y = int(topmost_y * scale)

        if bottommost_y <= topmost_y:
            image = image[:, bottommost_y:topmost_y, 0:s]
        else:
            image = image[:, topmost_y:bottommost_y, 0:s]

        # Convert tensor to numpy array
        np_array = image.numpy()

        # Reshape numpy array to (s, s, 3)
        np_array = np.transpose(np_array, (1, 2, 0))

        # Convert numpy array to OpenCV image
        # cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        # Convert numpy array to OpenCV grayscale image
        cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)

        # Reshape the image to (s, s)
        resized_img = cv2.resize(cv2_image, (size, size))

        out_path = os.path.abspath(image_ids[i]).replace(
            "KermanyV3", "KermanyV3_resized"
        )
        # print(out_path)
        cv2.imwrite(out_path, resized_img * 256)

        # Create a row of data
        new_row = {
            "path": image_ids[i],
            "shapex": size,
            "shapey": size,
            "x": 0,
            "y": int(bottommost_y * size / s),
            "width": size - 1,
            "height": int(topmost_y * size / s),
        }

        # Append the row to the DataFrame
        # result_df = result_df.append(new_row, ignore_index=True)

        new_row_df = pd.DataFrame([new_row])  # Create a DataFrame with a single row
        result_df = pd.concat([result_df, new_row_df], ignore_index=True)
