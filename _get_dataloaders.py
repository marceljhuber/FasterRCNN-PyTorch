import json
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

from _get_dataframes import process_data


# Utility Functions
def load_config(config_path="config.json"):
    """Loads the configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def list_jpeg_images(directory):
    """Returns a list of paths to JPEG images in the specified directory."""
    return [str(p) for p in Path(directory).rglob("*.jpg")]


def prepare_output_directories(out_dir, subdirs):
    """Ensures the required output subdirectories exist."""
    for subdir in subdirs:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)


# Dataset Classes
class OCTDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        img_path = records["path"].iloc[0]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])

        if self.transforms:
            # Apply PyTorch transforms (works with a dictionary)
            sample = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": target["labels"],
            }
            sample = self.transforms(sample)
            image = sample["image"]
            target["boxes"] = sample["bboxes"]
            target["labels"] = sample["labels"]

        return image, target, image_id

    def __len__(self):
        return len(self.image_ids)


class OCTTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe["image_id"].unique()[0:1000]
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

        # Preload images into memory
        self.images = {}
        print("Preloading images into memory...")
        for image_id in tqdm(self.image_ids, desc="Loading Images"):
            image_path = os.path.abspath(image_id)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.0  # Normalize to [0, 1]
                self.images[image_id] = image
            else:
                print(f"Warning: Unable to load image {image_id}")

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self.images[image_id]

        # Apply transformations
        if self.transforms:
            image = self.transforms(image)

        return image, image_id

    def __len__(self) -> int:
        return len(self.image_ids)



class RandomHorizontalFlipWithBoxes:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, boxes, labels = sample["image"], sample["bboxes"], sample["labels"]

        # Convert the NumPy array to a PIL image for transformation
        pil_image = Image.fromarray(
            (image * 255).astype(np.uint8)
        )  # Convert to [0, 255] range

        if torch.rand(1) < self.p:
            # Flip the image
            pil_image = F.hflip(pil_image)

            # Flip bounding boxes (adjust x-coordinates)
            width, _ = pil_image.size
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x_min and x_max

        # Convert the flipped PIL image back to NumPy array (or Tensor)
        image = (
            np.array(pil_image).astype(np.float32) / 255.0
        )  # Normalize back to [0, 1] range

        return {"image": image, "bboxes": boxes, "labels": labels}


class ToTensorWithBoxes:
    def __call__(self, sample):
        image, boxes, labels = sample["image"], sample["bboxes"], sample["labels"]
        image = F.to_tensor(image)  # Convert image to tensor
        return {"image": image, "bboxes": boxes, "labels": labels}


def get_train_transform():
    return T.Compose([RandomHorizontalFlipWithBoxes(p=0.5), ToTensorWithBoxes()])


def get_valid_transform():
    return T.Compose([ToTensorWithBoxes()])


def get_test_transform():
    return T.Compose([T.ToTensor()])


# Data Loaders
def get_data_loaders(
    inference: bool = False,
    data_dir=None,
    train_df: Optional[pd.DataFrame] = None,
    valid_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Creates DataLoader(s) for training/validation or testing.

    Parameters:
        inference (bool): If True, returns train and validation loaders; else test loader.
        train_df (pd.DataFrame): DataFrame for training data.
        valid_df (pd.DataFrame): DataFrame for validation data.
        test_df (pd.DataFrame): DataFrame for testing data.
        batch_size (int): Batch size for the loaders.
        num_workers (int): Number of workers for data loading.

    Returns:
        Union[Tuple[DataLoader, DataLoader], DataLoader]:
            - Tuple[train_loader, valid_loader] if `inference=True`.
            - test_loader if `inference=False`.
    """
    if not inference:
        if train_df is None or valid_df is None:
            raise ValueError(
                "train_df and valid_df must be provided for inference mode."
            )

        train_dataset = OCTDataset(train_df, data_dir, transforms=get_train_transform())
        valid_dataset = OCTDataset(valid_df, data_dir, transforms=get_valid_transform())

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        return train_loader, valid_loader
    else:
        if test_df is None:
            raise ValueError("test_df must be provided for testing mode.")

        test_dataset = OCTTestDataset(
            test_df, data_dir, transforms=get_test_transform()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        return test_loader


# Main Execution
if __name__ == "__main__":
    config = load_config()
    train_df, valid_df, test_df = process_data(config)
    train_loader, valid_loader = get_data_loaders(
        inference=True,
        data_dir=config["dir_train"],
        train_df=train_df,
        valid_df=valid_df,
        batch_size=4,
        num_workers=2,
    )

    test_loader = get_data_loaders(
        inference=False,
        data_dir=config["dir_train"],
        test_df=test_df,
        batch_size=1,
        num_workers=2,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
