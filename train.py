import datetime
import os
import time
import warnings
import json
import numpy as np
import torch
import torchvision
from numba import jit
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

warnings.filterwarnings("ignore")

from get_data_dfs import load_config, process_data
from get_data_loaders import get_data_loaders

# Load config and process data
config = load_config()

train_df, valid_df, test_df = process_data(config)
train_loader, valid_loader = get_data_loaders(
    inference=False,
    train_df=train_df,
    valid_df=valid_df,
    batch_size=8,
    num_workers=0,
)

class_to_int = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}

int_to_class = {v: k for k, v in class_to_int.items()}


def load_config(config_path="config.json"):
    """
    Loads the configuration from a JSON file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


config = load_config()

dir_train = config["dir_train"]
dir_test = config["dir_test"]
out_dir = config["out_dir"]
size = config["size"]
file = config["result_file"]

for subdir in ["0", "1", "2", "3"]:
    subdir_path = os.path.join(out_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

# Inputs required for training
model_path = os.path.join(os.getcwd(), "fasterrcnn_resnet50_fpn2_17July.pth")
early_stop = 3  # This is required for early stopping, the number of epochs we will wait with no improvement before stopping


# Load a pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    box_detections_per_img=1, pretrained=True
)


num_classes = 2  # 1 class (OCT) + background

# Get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


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


models = [model]
from ensemble_boxes import *


def make_ensemble_predictions(images):
    images = list(image.to(device) for image in images)
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Current device:", device)


# Configure the training parameters
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None

num_epochs = 10


do_train = True

if do_train:
    loss_hist = Averager()
    best_val = None
    patience = early_stop
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        itr = 1
        loss_hist.reset()
        model.train()
        for images, targets, image_ids in train_loader:
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if k == "labels" else v.float().to(device)
                    for k, v in t.items()
                }
                for t in targets
            ]  # [{k: v.double().to(device) if k =='boxes' else v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # Update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # At every epoch we will also calculate the validation IOU
        validation_image_precisions = []
        iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
        model.eval()
        for (
            images,
            targets,
            imageids,
        ) in valid_loader:  # return image, target, image_id
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if k == "labels" else v.float().to(device)
                    for k, v in t.items()
                }
                for t in targets
            ]

            predictions = make_ensemble_predictions(images)

            for i, image in enumerate(images):
                boxes, scores, labels = run_wbf(predictions, image_index=i)

                boxes = boxes.astype(np.int32).clip(min=0, max=1022553)

                preds = boxes
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = preds[preds_sorted_idx]
                gt_boxes = targets[i]["boxes"].cpu().numpy().astype(np.int32)
                image_precision = calculate_image_precision(
                    preds_sorted, gt_boxes, thresholds=iou_thresholds, form="coco"
                )

                validation_image_precisions.append(image_precision)
        val_iou = np.mean(validation_image_precisions)
        print(
            f"Epoch #{epoch+1} loss: {loss_hist.value}",
            "Validation IOU: {0:.4f}".format(val_iou),
            "Time taken :",
            str(datetime.timedelta(seconds=time.time() - start_time))[:7],
        )
        if not best_val:
            best_val = (
                val_iou  # So any validation roc_auc we have is the best one for now
            )
            print("Saving model")
            torch.save(model, model_path)  # Saving the model
            # continue
        if val_iou >= best_val:
            print("Saving model as IOU is increased from", best_val, "to", val_iou)
            best_val = val_iou
            patience = early_stop  # Resetting patience since we have new best validation accuracy
            torch.save(
                model, model_path
            )  # Saving current best model torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping. Best Validation IOU: {:.3f}".format(best_val))
                break
