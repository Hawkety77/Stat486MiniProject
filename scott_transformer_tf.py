import random
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras import ops

tf.get_logger().setLevel(logging.ERROR) # only log errors
tf.keras.utils.set_random_seed(9) # Joe Burrow

AUDIO_DATA_PATH = './Data/genres_original'
SPECT_DATA_PATH = './Data/images_original'
VALIDATION_SPLIT = 0.2

NUM_CLASSES = 10

BATCH_SIZE = 16
IMAGE_SIZE = 200 #72 # resize images to this size
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3) #128, 128, 3) # TODO: check input size
N_EPOCHS = 100

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001 # not implemented rn

## Transformer hyperparameters
TRANSFORMER_LAYERS = 2 #4
PATCH_SIZE = 6 # 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4 #4

transformer_units = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]  # Size of the transformer layers

mlp_head_units = [
    #256, #2048,
    128 #1024,
]  # Size of the dense layers of the final classifier

train = keras.utils.image_dataset_from_directory(
    SPECT_DATA_PATH,
    labels="inferred",
    batch_size=BATCH_SIZE,
    image_size=INPUT_SHAPE[0:2],
    shuffle=True,
    seed=9,
    validation_split=VALIDATION_SPLIT,
    subset='training')

validation = keras.utils.image_dataset_from_directory(
    SPECT_DATA_PATH,
    labels="inferred",
    batch_size=BATCH_SIZE,
    image_size=INPUT_SHAPE[0:2],
    shuffle=False,
    seed=9,
    validation_split=VALIDATION_SPLIT,
    subset='validation')

## Create dense layers
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

## Create patch embedding layer
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

## Encode patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )

        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

## Create transformer

def create_vit_classifier():
    inputs = keras.Input(shape=INPUT_SHAPE)

    # Create patches.
    patches = Patches(PATCH_SIZE)(inputs)

    # Encode patches.
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(NUM_CLASSES)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

## Compile and run model

# Compile, train, and evaluate the mode
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE #, weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    history = model.fit(
        train,
        validation_data=validation,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_split=0.1,
        #callbacks=[checkpoint_callback],
    )
    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)