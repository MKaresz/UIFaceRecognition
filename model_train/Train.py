import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# CONSTANTS
DATA_DIR = os.path.abspath("E:/EdgeTFtoAndroid/dataset_cache")
IMG_SIZE = 160
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
NUM_LABELS = 3  # e.g., [Male, Young, Smiling]
EPOCHS_BASE = 1 # ~10–20 is typical; stop earlier with EarlyStopping
EPOCHS_TUNE = 1 # ~8–20 typical
# model output decoding
ATTRS = ["Male", "Young", "Smiling"]


# inspired from the article:
# Content-Based Feature Extraction and Image Retrieval using Celeb-A dataset
# by Santosh Sawant
# https://www.linkedin.com/pulse/content-based-feature-extraction-image-retrieval-using-santosh-sawant


# HELPER FUCNTIONS
def _preprocess_image(image):
    """
    Resize and cast the image to float32
    
    We only cast to float here; actual MobileNetV2 normalization is applied 
    inside the model to keep preprocessing consistent between training and inference.
    """
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
    return tf.cast(image, tf.float32)


def _preprocess_label(attrs):
    """
    Convert selected attributes into a fixed-order (3) float32 vector.
    The order in ATTRS global variable, defines the meaning of each output neuron.
    """
    return tf.stack([tf.cast(attrs[name], tf.float32) for name in ATTRS], axis=-1)


def preprocess_for_model(example):
    """
    Apply both image and label preprocessing to a TFDS CelebA example.
    """
    image = _preprocess_image(example["image"])
    label = _preprocess_label(example["attributes"])
    return image, label


def make_pipeline(ds, training=False):
    """
    Pipeline for training; applying preprocessing, optional shuffling,
    batching, and prefetching.
    
    Shuffling is enabled only during training to avoid learning from the
    ordering of the dataset.

    Args:
        ds (tf.data.Dataset):
            A TFDS split (train/validation/test) containing raw CelebA
            samples.

        training (bool):
            If True, enables shuffling and training‑specific settings.

    Returns:
        tf.data.Dataset:
            Dataset yielding (image, label) batches ready for model training
            or for evaluation.
    """
    ds = ds.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)
    if training:
        dataset_size = ds.cardinality().numpy()
        buffer_size = min(50_000, dataset_size)
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def show_prediction(idx, model, threshold=0.5):
    """
    A Debugging fucntion, that displays model predictions for a specific
    test sample.

    This function retrieves a single preprocessed test example, runs it
    through the model, applies a decision threshold, and prints a comparison
    between predicted and true attribute values.

    Args:
        idx (int):
            Index of the test sample to inspect.

        model (tf.keras.Model):
            The trained model used for inference.

        threshold (float):
            Decision threshold applied to sigmoid outputs to obtain binary predictions.

    Prints:
        - True label vector
        - Binary prediction vector
        - Element‑wise correctness indicator
    """
    image, true_label = test_samples[idx]

    # Add batch dimension for inference
    pred = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]

    # Convert prediction to binary using threshold
    pred_binary = (pred > threshold).astype(np.float32)

    # True label as numpy
    true_label_np = true_label.numpy()

    # Element-wise comparison
    comparison = (pred_binary == true_label_np)

    print(f"\n=== Sample #{idx} ===")
    print("True label:           ", true_label_np)
    print("Prediction (binary):   ", pred_binary)
    print("Element-wise correct?: ", comparison)


# DATASET LOADER
(ds_train, ds_val, ds_test), info = tfds.load(
    "celeb_a:2.1.0",                     # you can omit ":2.1.0" since it's the default
    split=["train", "validation", "test"],
    with_info=True,
    as_supervised=False,
    data_dir=DATA_DIR,
    download=False,
    try_gcs=False
)

# Dataset splits
train_ds = make_pipeline(ds_train, training=True)
val_ds = make_pipeline(ds_val,   training=False)
test_ds = make_pipeline(ds_test,  training=False)

# Convert test dataset to an unbatched list of individual (image, label) pairs
test_samples = list(test_ds.unbatch().take(200)) # loads 200 images


## TRAINING
# Build model with a frozen base
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the pretrained backbone during warm‑up.
# This stabilizes training and allows the new classification head to adapt,
# before fine‑tuning deeper layers.
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # scale to [-1, 1]
x = base(x, training=False)  # keep BN layers in inference mode while base is frozen
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # small regularization
outputs = tf.keras.layers.Dense(NUM_LABELS, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(multi_label=True), tf.keras.metrics.BinaryAccuracy()]
)

# Warm-up training: only the head learns
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_BASE,
    callbacks=[ 
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", 
            mode="max", 
            patience=3, 
            restore_best_weights=True
        )
    ],
)

# Fine-tune: unfreeze the top of the backbone
base.trainable = True

# Keep BatchNorm layers frozen for stability (optional but recommended)
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Unfreeze only the top layers of the backbone ~30 layers
# Unfreezing too many layers increases overfitting risk and destroys
# pretrained weights.
for layer in base.layers[:-30]:
    layer.trainable = False

# Fine‑tuning requires a much smaller LR to avoid destroying pretrained weights.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),   # much smaller LR
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(multi_label=True), tf.keras.metrics.BinaryAccuracy()]
)

ft_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_TUNE,   # ~8–20 typical
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", 
                                             mode="max",
                                             factor=0.5, 
                                             patience=2, 
                                             min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", 
                                         mode="max",
                                         patience=3, 
                                         restore_best_weights=True)
    ],
)

# EVALUATE
print("Test evaluation:")
model.evaluate(test_ds)

# Inference example for simple testing the model
sample_batch = next(iter(test_ds))
preds = model.predict(sample_batch[0][:4])
for i, p in enumerate(preds):
    print(f"Sample {i}: " + ", ".join(f"{name}={p[j]:.3f}" for j, name in enumerate(ATTRS)))

# save model to TFlow Keras format
model.save("tflow_celeba_model.keras")


## Model Testing
loaded_model = tf.keras.models.load_model("tflow_celeba_model.keras")

## test KERAS model
for i in range(42):
    show_prediction(i, loaded_model)


## MODEL CONVERTER to TFlite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# tf.float32 format, weight storage only; computation remains in float32.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
f32_tflite_model = converter.convert()
# Save the converted model
with open ('f32_celeba_model.tflite', 'wb') as f:
    f.write(f32_tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
# tf.float16 for weight storage only; computation remains in float32.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
f16_tflite_model = converter.convert()
# Save the converted model
with open ('f16_celeba_model.tflite', 'wb') as f:
    f.write(f16_tflite_model)

## Int8 Quantization -> model's int8 tensors will use -128 to 127
# Int8 requires a representative dataset to calibrate activation ranges; 100~300 images
def representative_dataset():
    """
    Generator that yields representative samples for INT8 post‑training quantization.

    Using samples to estimate activation ranges the representative dataset
    should reflect the distribution of real inputs. (100–300 samples)

    Yields:
        list[tf.Tensor]:
            A single‑image batch (shape: (1, IMG_SIZE, IMG_SIZE, 3)) for
            TFLite calibration.
    """
    # Build a small calibration set from the *raw* TFDS split (not the batched train_ds)
    calib_ds = (
        ds_train
        .map(preprocess_for_model, num_parallel_calls=AUTOTUNE)  # (image_preprocessed, label)
        .map(lambda img, lbl: img)                      # keep images only
        .batch(1)                                       # (1, IMG_SIZE, IMG_SIZE, 3)
        .take(200)                                      # 100–300 samples is typical
        .prefetch(AUTOTUNE)
    )
    for batch in calib_ds:
        # TFLite accepts Tensors or NumPy arrays; both are fine
        yield [batch]


# Apply during conversion
i8q_converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
i8q_converter.optimizations = [tf.lite.Optimize.DEFAULT]
i8q_converter.representative_dataset = representative_dataset
i8q_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Keep I/O in float32 to standardize inference code for every model!
# Internally the model runs in int8, but model is feeded with float32 tensors
#i8q_converter.inference_input_type = tf.int8
#i8q_converter.inference_output_type = tf.float32


# INT8 I/O: the representative data uses preprocessed [-1,1] inputs.
i8q_tflite_model = i8q_converter.convert()
with open ('i8q_celeba_model.tflite', 'wb') as f:
    f.write(i8q_tflite_model)

# inference test: check model runs, output shape and values
# are correct
image, _ = test_samples[0]
image = tf.expand_dims(image, axis=0)  # shape (1, 160, 160, 3)
