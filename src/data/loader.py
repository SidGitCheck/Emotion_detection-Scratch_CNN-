
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # For a nice progress bar
import os
import sys

# --- Constants ---
# We define constants at the top. This is a best practice.
# It makes the code easy to read and modify.
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 7
DATA_PATH = 'data/fer2013.csv' # Path from the project root

def _parse_data_from_df(df):
    """
    Private helper function to parse pixel strings and labels from a DataFrame.
    The underscore '_' in the name signals this is an "internal" function.
    """
    
    # Get all pixel strings from the DataFrame
    pixel_data = df['pixels'].tolist()
    
    images = []
    # Use tqdm to show a progress bar. Parsing 28,000+ strings takes a moment.
    print(f"Parsing {len(pixel_data)} pixel strings...")
    for pixels in tqdm(pixel_data):
        # 1. Split the string by space
        # 2. Convert each value to an integer (uint8 is 0-255)
        # 3. Reshape into a 48x48 image
        img = np.array(pixels.split(), dtype=np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH)
        images.append(img)
        
    # Convert list of images to a single 4D NumPy array
    images = np.array(images)
    
    # --- Preprocessing ---
    # 1. Convert to float32 for (0-1) normalization
    images = images.astype('float32') / 255.0
    
    # 2. Expand dimensions to (N, 48, 48, 1)
    # The CNN needs a "channel" dimension, which is 1 for grayscale.
    images = np.expand_dims(images, -1)
    
    # --- Labels ---
    # 1. Get the emotion labels
    labels = df['emotion'].values
    
    # 2. One-hot encode the labels
    if labels is not None:
        labels = to_categorical(labels, num_classes=NUM_CLASSES)
    
    return images, labels

def load_fer2013(csv_path=DATA_PATH):
    """
    Main public function to load and preprocess the FER-2013 dataset.
    
    This is the function our `train.py` will import and call.
    
    Args:
        csv_path (str): Path to the fer2013.csv file.
        
    Returns:
        Tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        # Add a helpful error message
        print(f"Error: File not found at {csv_path}")
        print("Please download 'fer2013.csv' from Kaggle and place it in the 'data/' directory.")
        sys.exit(1) # Exit the script with an error
        
    df = pd.read_csv(csv_path)
    
    # --- Split Data ---
    # The 'Usage' column defines the dataset split
    df_train = df[df['Usage'] == 'Training'].copy()
    df_val = df[df['Usage'] == 'PublicTest'].copy()
    df_test = df[df['Usage'] == 'PrivateTest'].copy()
    
    print(f"Found {len(df_train)} training, {len(df_val)} validation, and {len(df_test)} test samples.")

    # --- Parse Data ---
    print("\nProcessing Training Data...")
    X_train, y_train = _parse_data_from_df(df_train)
    
    print("\nProcessing Validation Data...")
    X_val, y_val = _parse_data_from_df(df_val)
    
    print("\nProcessing Test Data...")
    X_test, y_test = _parse_data_from_df(df_test)
    
    print("\nData loading complete.")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --- This is the "Test" block ---
# If we run `python src/data/loader.py` directly, this code will execute.
# This is how we test this module on its own.
if __name__ == '__main__':
    print("--- Running Data Loader Standalone Test ---")
    
    # We call our main function
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013()
    
    print("\n--- Test Results ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    print(f"Image data type: {X_train.dtype}")
    print(f"Image min value: {np.min(X_train)}")
    print(f"Image max value: {np.max(X_train)}")
    print(f"Label sample (one-hot): {y_train[0]}")
    print("--- Standalone Test Complete ---")