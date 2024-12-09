import os
import warnings
from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

from _get_dataframes import load_config, process_data
from _get_dataloaders import get_data_loaders

# Load config and process data
config = load_config()


train_df, valid_df, test_df = process_data(config)
test_loader = get_data_loaders(
    inference=True,
    test_df=test_df,
    batch_size=1,
    num_workers=0,
)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

dir_train = os.path.join(parent_dir, config["dir_train"])
dir_test = os.path.join(parent_dir, config["dir_test"])
out_dir = os.path.join(parent_dir, config["out_dir"])

size = config["size"]
file = config["result_file"]

int_to_class = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}

class_to_int = {v: k for k, v in int_to_class.items()}

for subdir in ["0", "1", "2", "3"]:
    subdir_path = os.path.join(out_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

# Inputs required for training
model_path = os.path.join(os.getcwd(), "fasterrcnn_resnet50_fpn2_17July.pth")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Current device:", device)


model = torch.load(model_path)
torch.save(model, model_path)


class OCTTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]

        # Construct the full image path (assuming image_id is the filename)
        image_path = f"{self.image_dir}/{image_id}"
        image_path = image_id

        # Read image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= float(255)  # Normalize to [0, 1] if you want

        # Apply transformations
        if self.transforms:
            image = self.transforms(image)  # Transform directly on the image

        return image, image_id

    def __len__(self) -> int:
        return len(self.image_ids)


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


from ensemble_boxes import *


def make_predictions(images):
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


# Cre
torch.cuda.empty_cache()

detection_threshold = 0.5
results = []
outputs = []
resize_transform = transforms.Resize((size, size))

for images, image_ids in tqdm(test_loader):
    images = list(image.to(device) for image in images)
    predictions = make_predictions(images)

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
        cv2_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

        # Reshape the image to (s, s)
        resized_img = cv2.resize(cv2_image, (size, size))

        # Using pathlib for better path handling
        image_name = Path(image_ids[i]).stem  # This gets the filename without extension
        image_class = class_to_int[image_name.split("-")[0]]
        # out_path = Path(out_dir) / str(image_class) / f"{image_name}.png"
        out_path = Path(out_dir) / str(image_class) / f"{image_name}.jpeg"

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

        new_row_df = pd.DataFrame([new_row])  # Create a DataFrame with a single row
        result_df = pd.concat([result_df, new_row_df], ignore_index=True)

