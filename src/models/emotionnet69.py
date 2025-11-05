
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add, 
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def conv_bn_swish(inputs, filters, kernel_size, strides=(1, 1)):
    """A helper for a standard Conv->BN->Swish block."""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    return x

def residual_bottleneck_block(inputs, filters, strides=(1, 1)):
    """
    Creates a residual "bottleneck" block.
    
    This is the efficient block from modern ResNets.
    1x1 Conv (Squeeze) -> 3x3 Conv -> 1x1 Conv (Expand)
    """
    shortcut = inputs
    input_channels = inputs.shape[-1]
    
    # --- Main Path ---
    # Squeeze: 1x1 conv to reduce channels
    x = conv_bn_swish(inputs, filters // 4, kernel_size=(1, 1), strides=strides)
    
    # Learn: 3x3 conv for spatial features
    x = conv_bn_swish(x, filters // 4, kernel_size=(3, 3))
    
    # Expand: 1x1 conv to restore channels
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # --- Shortcut Path ---
    # If we downsampled or changed filters, we must project the shortcut
    if strides != (1, 1) or input_channels != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut)
        
    # --- Add & Activate ---
    x = Add()([shortcut, x])
    x = Activation('swish')(x)
    return x

def build_layer(inputs, filters, blocks, strides=(1, 1)):
    """Helper to stack multiple residual blocks."""
    # The first block handles downsampling (if strides > 1)
    x = residual_bottleneck_block(inputs, filters, strides=strides)
    # Subsequent blocks just learn
    for _ in range(1, blocks):
        x = residual_bottleneck_block(x, filters)
    return x

# --- Main Model: EmotionNet-69 (Lightweight 1.8M Version) ---
def EmotionNet69(input_shape=(48, 48, 1), num_classes=7, l2_reg=1e-5):
    """
    Creates the EmotionNet-69 model architecture (~1.8M params).
    Uses efficient bottleneck residual blocks.
    """
    regularizer = l2(l2_reg)
    
    inputs = Input(shape=input_shape)

    # --- Entry Flow (Stem) ---
    # 48x48x1 -> 24x24x64
    x = conv_bn_swish(inputs, 64, kernel_size=(7, 7), strides=(2, 2))
    
    # --- Main Residual Blocks (ResNet-style) ---
    # 24x24x64 -> 24x24x64
    x = build_layer(x, 64, blocks=2)
    
    # 24x24x64 -> 12x12x128
    x = build_layer(x, 128, blocks=2, strides=(2, 2))
    
    # 12x12x128 -> 6x6x256
    x = build_layer(x, 256, blocks=2, strides=(2, 2))
    
    # 6x6x256 -> 3x3x512
    x = build_layer(x, 512, blocks=2, strides=(2, 2))

    # --- Exit Flow (Head) ---
    # 3x3x512 -> 512
    x = GlobalAveragePooling2D()(x)
    
    # --- Classifier ---
    # This dense layer is the new bottleneck for parameters
    x = Dense(256, activation='swish', kernel_regularizer=regularizer)(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- This is the "Test" block ---
# If we run `python src/models/emotionnet69.py` directly, this code will execute.
if __name__ == '__main__':
    print("--- Running Model Architecture Standalone Test (v3 - Bottleneck) ---")
    
    # Build the model
    model = EmotionNet69()
    
    # Print the model summary
    print(model.summary())
    
    print("\nModel built successfully!")
    print(f"Total parameters: {model.count_params()}")
    print("--- Standalone Test Complete ---")