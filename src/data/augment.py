
import tensorflow as tf
import numpy as np

# --- CutMix Helper Function ---

def _rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    
    Args:
        size (tuple): The (height, width) of the image.
        lam (float): The lambda value (mixing proportion).
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box.
    """
    H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate box coordinates, clamping them to be within image bounds
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# --- Main Augmentation Functions ---

def mixup(images, labels, alpha=0.4):
    """
    Applies MixUp augmentation to a batch of images and labels.
    
    Args:
        images (np.ndarray): Batch of images (N, H, W, C).
        labels (np.ndarray): Batch of one-hot labels (N, num_classes).
        alpha (float): Beta distribution parameter. Controls the mixing strength.
        
    Returns:
        (np.ndarray, np.ndarray): The mixed images and labels.
    """
    batch_size = images.shape[0]
    
    # 1. Generate lambda (mixing proportion) from a Beta distribution
    lam = np.random.beta(alpha, alpha, batch_size)
    
    # 2. Reshape lambda for broadcasting
    # (N,) -> (N, 1, 1, 1) for images
    # (N,) -> (N, 1) for labels
    lam_images = lam.reshape(batch_size, 1, 1, 1)
    lam_labels = lam.reshape(batch_size, 1)

    # 3. Create a shuffled batch
    # This is how we get the "second" image (image B) to mix with
    indices = np.random.permutation(batch_size)
    images_b = images[indices]
    labels_b = labels[indices]
    
    # 4. Perform the MixUp
    # mixed_img = lam * img_A + (1 - lam) * img_B
    # mixed_label = lam * label_A + (1 - lam) * label_B
    mixed_images = lam_images * images + (1. - lam_images) * images_b
    mixed_labels = lam_labels * labels + (1. - lam_labels) * labels_b
    
    return mixed_images, mixed_labels

def cutmix(images, labels, alpha=1.0):
    """
    Applies CutMix augmentation to a batch of images and labels.
    
    Args:
        images (np.ndarray): Batch of images (N, H, W, C).
        labels (np.ndarray): Batch of one-hot labels (N, num_classes).
        alpha (float): Beta distribution parameter.
        
    Returns:
        (np.ndarray, np.ndarray): The CutMixed images and labels.
    """
    batch_size, H, W, _ = images.shape
    
    # 1. Create a shuffled batch
    indices = np.random.permutation(batch_size)
    images_b = images[indices]
    labels_b = labels[indices]

    # 2. Generate lambda (mixing proportion)
    # For CutMix, lam represents the area of the patch
    lam = np.random.beta(alpha, alpha)
    
    # 3. Generate the bounding box
    bbx1, bby1, bbx2, bby2 = _rand_bbox((H, W), lam)
    
    # 4. Create a copy of the original images to modify
    mixed_images = np.copy(images)
    
    # 5. Paste the patch from image B onto image A
    mixed_images[:, bby1:bby2, bbx1:bbx2, :] = images_b[:, bby1:bby2, bbx1:bbx2, :]
    
    # 6. Adjust lambda to be the *actual* area of the patch
    # This is important if the box was clipped at the edge
    area_patch = (bbx2 - bbx1) * (bby2 - bby1)
    area_total = W * H
    lam_adjusted = area_patch / area_total

    # 7. Mix the labels based on the patch area
    # mixed_label = (1 - lam_adjusted) * label_A + lam_adjusted * label_B
    mixed_labels = (1. - lam_adjusted) * labels + lam_adjusted * labels_b
    
    return mixed_images, mixed_labels


# --- This is the "Test" block ---
# If we run `python src/data/augment.py` directly, this code will execute.
if __name__ == '__main__':
    print("--- Running Augmentation Standalone Test ---")
    
    # We need to install these to run the test
    try:
        import matplotlib.pyplot as plt
        import cv2
    except ImportError:
        print("\nPlease install 'matplotlib' and 'opencv-python' to run this test:")
        print("pip install matplotlib opencv-python")
        sys.exit(1)
        
    # We'll re-use our data loader to get some test images!
    # This is why modular code is so good.
    from src.data.loader import load_fer2013
    
    print("Loading a small batch of data for testing...")
    # We only load the training data. We don't need val/test here.
    (X_train, y_train), _, _ = load_fer2013()
    
    # Get a small batch of 4 images
    sample_images = X_train[:4]
    sample_labels = y_train[:4]
    print(f"Original image shape: {sample_images.shape}")
    
    # --- Test MixUp ---
    print("\nTesting MixUp...")
    mixed_up_images, mixed_up_labels = mixup(sample_images, sample_labels)
    
    # --- Test CutMix ---
    print("Testing CutMix...")
    cutmix_images, cutmix_labels = cutmix(sample_images, sample_labels)

    print("\nTest complete. Displaying results...")
    
    # --- Visualization ---
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Augmentation Test (MixUp & CutMix)", fontsize=16)

    # Set titles for rows
    axes[0, 0].set_ylabel("Original", rotation=90, size='large')
    axes[1, 0].set_ylabel("MixUp", rotation=90, size='large')
    axes[2, 0].set_ylabel("CutMix", rotation=90, size='large')
    
    for i in range(4):
        # Original
        axes[0, i].imshow(sample_images[i], cmap='gray')
        axes[0, i].set_title(f"Label: {np.argmax(sample_labels[i])}")
        axes[0, i].axis('off')
        
        # MixUp
        axes[1, i].imshow(mixed_up_images[i], cmap='gray')
        axes[1, i].set_title(f"Mixed Label (Top 2)\n{np.argsort(mixed_up_labels[i])[-2:]}")
        axes[1, i].axis('off')
        
        # CutMix
        axes[2, i].imshow(cutmix_images[i], cmap='gray')
        axes[2, i].set_title(f"Mixed Label (Top 2)\n{np.argsort(cutmix_labels[i])[-2:]}")
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()