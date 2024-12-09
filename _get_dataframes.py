import os
import numpy as np
import pandas as pd
import json
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def list_jpeg_images(directory):
    """
    Returns a list of paths to JPEG images in the specified directory.
    """
    jpeg_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpeg_images.append(os.path.join(root, file))
    return jpeg_images

def load_config(config_path="config.json"):
    """
    Loads the configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def process_data(config):
    """
    Processes training, validation, and test data based on the provided config.
    Returns the train, validation, and test dataframes.
    """
    # Load directories and settings from config
    dir_train = config["dir_train"]
    dir_test = config["dir_test"]
    out_dir = config["out_dir"]
    size = config["size"]
    file = config["result_file"]

    # Create subdirectories for output
    for subdir in ['0', '1', '2', '3']:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    # Get image paths
    images_path_train = list_jpeg_images(dir_train)
    images_path_test = list_jpeg_images(dir_test)

    # Load and process training images
    images_im_train = [Image.open(filename) for filename in images_path_train]

    # Load annotations from the JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    # Initialize dataframe
    my_columns = ['image_id', 'path', 'shapex', 'shapey', 'x', 'y', 'w', 'h']
    my_data = {col: [] for col in my_columns}
    my_dataframe = pd.DataFrame(data=my_data)

    # Populate image_ids
    my_dataframe['image_id'] = [img['id'] for img in data['images']]

    # Process bounding box annotations
    for annot in data['annotations']:
        cur_image_id = annot['image_id']
        cur_image_rows = my_dataframe['image_id'] == cur_image_id

        my_dataframe.loc[cur_image_rows, ['x', 'y', 'w', 'h']] = [int(coord) for coord in annot['bbox']]

    # Create dictionaries for converting the classes to their numerical ID
    class_to_int = {
        0: "CNV",
        1: "DME",
        2: "DRUSEN",
        3: "NORMAL"
    }
    int_to_class = {v: k for k, v in class_to_int.items()}

    # Process image file paths and shape
    for annot in data['images']:
        cur_image_id = annot['id']
        image_filename = annot['file_name'][16:].split('-')[0]
        image_class = int_to_class[image_filename]

        my_dataframe.loc[cur_image_id, 'path'] = os.path.join(dir_train, str(image_class), annot['file_name'][16:])
        my_dataframe.loc[cur_image_id, ['shapex', 'shapey']] = [int(annot['width']), int(annot['height'])]

    # Remove rows with missing values and convert to appropriate data types
    my_dataframe = my_dataframe.dropna()
    my_dataframe[['shapex', 'shapey', 'x', 'y', 'w', 'h']] = my_dataframe[['shapex', 'shapey', 'x', 'y', 'w', 'h']].astype(int)
    my_dataframe['path'] = my_dataframe['path'].astype(str)

    # Scale bounding box coordinates to target size
    scalx = my_dataframe['shapex'] / size
    scaly = my_dataframe['shapey'] / size

    # Divide each coordinate by its corresponding scaling factor
    my_dataframe['x'] /= scalx
    my_dataframe['y'] /= scaly
    my_dataframe['w'] /= scalx
    my_dataframe['h'] /= scaly

    # Ensure integer types for bounding box columns
    my_dataframe[['shapex', 'shapey', 'x', 'y', 'w', 'h']] = my_dataframe[['shapex', 'shapey', 'x', 'y', 'w', 'h']].astype(int)

    train_df = my_dataframe.copy()

    # Convert bounding box columns to float64 for model compatibility
    train_df[['x', 'y', 'w', 'h']] = train_df[['x', 'y', 'w', 'h']].astype(np.float64)

    # Split data into train and validation sets
    threshold = int(len(train_df['x']) * 0.2)
    image_ids = train_df['image_id'].unique()

    # Shuffle and split dataset
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_ids = image_ids[-threshold:]
    train_ids = image_ids[:-threshold]

    # Prepare validation and training dataframes
    valid_df = train_df[train_df['image_id'].isin(valid_ids)].reset_index(drop=True)
    train_df = train_df[train_df['image_id'].isin(train_ids)].reset_index(drop=True)

    # Prepare test dataframe
    test_df = pd.DataFrame()
    test_df['image_id'] = images_path_test

    return train_df, valid_df, test_df

# Entry point for direct execution
if __name__ == "__main__":
    # Load config and process data
    config = load_config()
    train_df, valid_df, test_df = process_data(config)

    # Print the resulting sizes of the dataframes
    print(f"Training set size: {train_df.shape}, Validation set size: {valid_df.shape}, Test set size: {test_df.shape}")

    # Optionally, save dataframes to CSV for inspection
    # train_df.to_csv("train_df.csv", index=False)
    # valid_df.to_csv("valid_df.csv", index=False)
    # test_df.to_csv("test_df.csv", index=False)