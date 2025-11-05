
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
import pickle
from sklearn.utils import class_weight

# --- Import Our Custom Modules ---
# We can do this because our (venv) and project root are set up.
from src.data.loader import load_fer2013
from src.data.augment import mixup, cutmix
from src.models.emotionnet69 import EmotionNet69

# --- Constants & Configuration ---
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 7

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # This is also needed for some TF ops
    tf.keras.utils.set_random_seed(seed)
    print(f"Random seed set to {seed}")

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train EmotionNet-69 model.")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models and history (default: models)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Alpha for MixUp (default: 0.4)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='Alpha for CutMix (default: 1.0)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (default: 25)')
    
    return parser.parse_args()

def build_augmentation_pipeline():
    """Creates a Keras Sequential model for standard augmentation."""
    # This pipeline will run on the GPU, making it very fast.
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.1, fill_mode='nearest'),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
        tf.keras.layers.RandomFlip(mode='horizontal'),
    ])

def preprocess_data(images, labels, augment_pipeline, mixup_alpha, cutmix_alpha, training=True):
    """
    Applies augmentations (standard, MixUp, CutMix) to a batch.
    This function is designed to be used with `tf.data.Dataset.map()`.
    """
    # --- NEW FIX: ADD THIS LINE ---
    # This forces the "else" branch to be float32, matching the other branches.
    labels = tf.cast(labels, tf.float32)
    
    if not training:
        return images, labels

    # Apply standard augmentations (Rotation, Flip, etc.)
    images = augment_pipeline(images, training=True)
    
    # --- Advanced Augmentation (MixUp/CutMix) ---
    # We use tf.py_function to wrap our numpy-based augment.py functions
    # This is the "glue" that lets our NumPy code work in a tf.data pipeline.
    
    def _apply_mixup(images, labels):
        return mixup(images, labels, alpha=mixup_alpha)
    
    def _apply_cutmix(images, labels):
        return cutmix(images, labels, alpha=cutmix_alpha)

    # Randomly choose between MixUp, CutMix, or just standard augmentation
    aug_choice = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    
    if aug_choice == 0:
        # Apply MixUp
        images, labels = tf.py_function(
            _apply_mixup, 
            [images, labels], 
            [tf.float32, tf.float32]
        )
    elif aug_choice == 1:
        # Apply CutMix
        images, labels = tf.py_function(
            _apply_cutmix, 
            [images, labels], 
            [tf.float32, tf.float32]
        )
    # else (aug_choice == 2): Do nothing, just use standard augmentations
        
    # Ensure shapes are set correctly after py_function
    images.set_shape([None, IMG_HEIGHT, IMG_WIDTH, 1])
    labels.set_shape([None, NUM_CLASSES])
    
    return images, labels


def main(args):
    """Main training function."""
    
    # 1. Set Seed & Create Dirs
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 2. Load Data
    print("Loading FER-2013 data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013()
    print("Data loaded successfully.")
    
    # 3. Calculate Class Weights (to handle imbalance)
    print("Calculating class weights...")
    # We need to get the "class index" (0, 1, 2...) from the one-hot labels
    y_train_indices = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    # Convert to a dictionary for Keras
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights_dict}")

    # 4. Create tf.data.Datasets
    print("Creating tf.data pipelines...")
    augment_pipeline = build_augmentation_pipeline()
    
    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(
        lambda img, lbl: preprocess_data(
            img, lbl, augment_pipeline, args.mixup_alpha, args.cutmix_alpha, training=True
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Validation dataset (no shuffle, no augmentation)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(args.batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 5. Build & Compile Model
    print("Building model...")
    model = EmotionNet69(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES)
    
    # Cosine LR Scheduler
    total_steps = (len(X_train) // args.batch_size) * args.epochs
    cosine_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=0.01 # Go down to 1% of initial LR
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_lr_schedule)
    
    # Loss with Label Smoothing
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    model.summary()

    # 6. Define Callbacks
    print("Setting up callbacks...")
    # Save the best model based on validation accuracy
    best_model_path = os.path.join(args.model_dir, f'emotionnet69_seed{args.seed}_best.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Stop training if val_accuracy doesn't improve
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    # Log to TensorBoard (optional, but good practice)
    log_dir = os.path.join('logs', f'seed_{args.seed}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, early_stopping, tensorboard_callback]

    # 7. Start Training
    print(f"\n--- Starting Training (Seed: {args.seed}) ---")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Initial LR: {args.lr}")
    
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        class_weight=class_weights_dict
    )

    # 8. Save Final Model and History
    print("\nTraining complete.")
    final_model_path = os.path.join(args.model_dir, f'emotionnet69_seed{args.seed}_final.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    history_path = os.path.join(args.model_dir, f'history_seed{args.seed}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")

# --- Run the script ---
if __name__ == '__main__':
    args = get_args()
    main(args)