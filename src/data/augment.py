
import tensorflow as tf
import numpy as np
import sys

# --- CutMix Helper Function ---

def _rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
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
    """
    # --- FIX ---
    # Convert Tensors to NumPy arrays
    images_np = images.numpy()
    labels_np = labels.numpy()
    
    batch_size = images_np.shape[0]
    
    # 1. Generate lambda (mixing proportion)
    lam = np.random.beta(alpha, alpha, batch_size)
    
    # 2. Reshape lambda for broadcasting
    lam_images = lam.reshape(batch_size, 1, 1, 1)
    lam_labels = lam.reshape(batch_size, 1)

    # 3. Create a shuffled batch
    indices = np.random.permutation(batch_size)
    images_b = images_np[indices]  # Use numpy arrays
    labels_b = labels_np[indices]  # Use numpy arrays
    
    # 4. Perform the MixUp
    mixed_images = lam_images * images_np + (1. - lam_images) * images_b
    mixed_labels = lam_labels * labels_np + (1. - lam_labels) * labels_b
    
    # Cast back to float32
    return mixed_images.astype(np.float32), mixed_labels.astype(np.float32)

def cutmix(images, labels, alpha=1.0):
    """
    Applies CutMix augmentation to a batch of images and labels.
    """
    # --- FIX ---
    # Convert Tensors to NumPy arrays
    images_np = images.numpy()
    labels_np = labels.numpy()
    
    batch_size, H, W, _ = images_np.shape
    
    # 1. Create a shuffled batch
    indices = np.random.permutation(batch_size)
    images_b = images_np[indices]  # Use numpy arrays
    labels_b = labels_np[indices]  # Use numpy arrays

    # 2. Generate lambda (mixing proportion)
    lam = np.random.beta(alpha, alpha)
    
    # 3. Generate the bounding box
    bbx1, bby1, bbx2, bby2 = _rand_bbox((H, W), lam)
    
    # 4. Create a copy of the original images to modify
    mixed_images = np.copy(images_np)
    
    # 5. Paste the patch from image B onto image A
    mixed_images[:, bby1:bby2, bbx1:bbx2, :] = images_b[:, bby1:bby2, bbx1:bbx2, :]
    
    # 6. Adjust lambda to be the *actual* area of the patch
    area_patch = (bbx2 - bbx1) * (bby2 - bby1)
    area_total = W * H
    lam_adjusted = area_patch / area_total

    # 7. Mix the labels based on the patch area
    mixed_labels = (1. - lam_adjusted) * labels_np + lam_adjusted * labels_b
    
    # Cast back to float32
    return mixed_images.astype(np.float32), mixed_labels.astype(np.float32)


# --- This is the "Test" block ---
if __name__ == '__main__':
    print("--- Running Augmentation Standalone Test ---")
    
    try:
        import matplotlib.pyplot as plt
        import cv2
    except ImportError:
        print("\nPlease install 'matplotlib' and 'opencv-python' to run this test:")
        print("pip install matplotlib opencv-python")
        sys.exit(1)
        
    # We must also disable eager execution for this test now
    tf.compat.v1.disable_eager_execution()
    
    from src.data.loader import load_fer2013
    
    print("Loading a small batch of data for testing...")
    (X_train, y_train), _, _ = load_fer2013()
    
    sample_images = X_train[:4]
    sample_labels = y_train[:4]
    print(f"Original image shape: {sample_images.shape}")
    
    # We need to wrap our functions in tf.function to test
    # but for simplicity, we'll just test the numpy part
    
    print("\nTesting MixUp...")
    # Manually create tensors to simulate the tf.data pipeline
    images_tensor = tf.convert_to_tensor(sample_images)
    labels_tensor = tf.convert_to_tensor(sample_labels, dtype=tf.float32)
    
    mixed_up_images, mixed_up_labels = mixup(images_tensor, labels_tensor)
    print(f"MixUp labels dtype: {mixed_up_labels.dtype}") # Should be float32
    
    print("Testing CutMix...")
    cutmix_images, cutmix_labels = cutmix(images_tensor, labels_tensor)
    print(f"CutMix labels dtype: {cutmix_labels.dtype}") # Should be float32

    print("\nTest complete. Displaying results...")
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Augmentation Test (MixUp & CutMix)", fontsize=16)

    axes[0, 0].set_ylabel("Original", rotation=90, size='large')
    axes[1, 0].set_ylabel("MixUp", rotation=90, size='large')
    axes[2, 0].set_ylabel("CutMix", rotation=90, size='large')
    
    for i in range(4):
        axes[0, i].imshow(sample_images[i], cmap='gray')
        axes[0, i].set_title(f"Label: {np.argmax(sample_labels[i])}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mixed_up_images[i], cmap='gray')
        axes[1, i].set_title(f"Mixed Label (Top 2)\n{np.argsort(mixed_up_labels[i])[-2:]}")
        axes[1, i].axis('off')
        
        axes[2, i].imshow(cutmix_images[i], cmap='gray')
        axes[2, i].set_title(f"Mixed Label (Top 2)\n{np.argsort(cutmix_labels[i])[-2:]}")
        axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()