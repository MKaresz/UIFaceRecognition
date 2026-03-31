import csv
import os
import json
import time
from typing import Tuple, Dict

import tensorflow as tf
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from typing import Any

MODELS = [
    "tflow_celeba_model.keras",
    "i8q_celeba_model.tflite",
    "f16_celeba_model.tflite",
    "f32_celeba_model.tflite",
]
MODELS_MAPPING = {i: model for i, model in enumerate(MODELS, start=1)}
RUN_DEBUG = True


# Shared preprocessing
def preprocess_image(img_path: str, target_size: Tuple[int, int]=(160, 160)) -> np.ndarray:
    """
    Load and preprocess an image for model inference.

    Args:
        img_path: path to image
        target_size: desired width, height of the output image

     Returns:
        A 4D array of shape (1, height, width, channels) suitable for
        feeding into TensorFlow or TFLite models.
    """
    img = Image.open(img_path).resize(target_size)
    arr = np.array(img).astype(np.float32)

    arr = tf.cast(arr, tf.float32)
    arr = np.expand_dims(arr, axis=0) # NHWC
    return arr


# Load model
def load_tflite_model(model_path: str) -> Interpreter:
    """
    Load and initialize a TFLite model interpreter.

    Args:
        model_path: Path to the `.tflite` model file.

    Returns:
        A fully initialized TFLite Interpreter instance ready for inference.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # warm up the model - run with dummy input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    dummy_input = np.zeros(input_shape, dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    return interpreter


def load_keras_model(model_path: str) -> Any:
    """
    Load and warm up a Keras model for inference and performs a
    warm‑up inference using a zero‑initialized dummy input. Warming up helps
    reduce latency for the first real prediction call.

    Args:
       model_path: Path to the `.keras` model file.

    Returns:
       A loaded and warmed‑up Keras model instance ready for inference.
       """
    keras_model = tf.keras.models.load_model(model_path)

    # Warme up - run one dummy inference
    dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
    keras_model.predict(dummy)

    return keras_model

# Measure model performance
def run_tflite_model(img_arr: np.ndarray,  interpreter: Interpreter) -> Tuple[np.ndarray, float]:
    """
    Run inference on a TFLite model and measure execution time in milliseconds
    using a high‑precision performance counter.

    Args:
        img_arr: Preprocessed input image array in NHWC format - (1, height, width, channels)
        interpreter: A TensorFlow Lite Interpreter instance  with allocated tensors.

    Returns:
        - The model output as a NumPy array.
        - The inference time in milliseconds.
        """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_arr)

    start = time.perf_counter()
    interpreter.invoke()
    end = time.perf_counter()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    inference_time = (end - start) * 1000.0  # ms

    return output, inference_time


def run_keras_model(img_arr: np.ndarray, model: tf.keras.Model) -> Tuple[np.ndarray, float]:
    """Run inference on a Keras model and measure execution time in milliseconds
    using a high‑precision performance counter.

    Args:
        img_arr: Preprocessed input image array in NHWC format - (1, height, width, channels).
        model: A loaded and ready‑to‑use Keras model.

    Returns:
        - The model output as a NumPy array.
        - The inference time in milliseconds.
        """
    start = time.perf_counter()
    output = model.predict(img_arr, verbose=0)[0]
    end = time.perf_counter()

    inference_time = (end - start) * 1000.0  # milliseconds

    return output, inference_time


# Evaluate model
def evaluate_model(model: Any, is_keras: bool, eval_dir: str="eval_set") -> Dict[str, float]:
    """
    Evaluate a model by running on all images in eval_set folder and return the
    results including the ground truth labels..

    Args>
        model: loaded model to evaluate
        is_keras:  If True, the model is treated as a Keras model
        eval_dir: a folder that contains a set of images for evaluation, default is eval_set.

     Returns:
        A dictionary mapping each filename to its evaluation results, including:
            - model predictions for Male/Young/Smile
            - inference time in milliseconds
            - ground‑truth labels for each category
    """
    with open("eval_ground_truth.json", "r") as f:
        ground_truth = json.load(f)
        images = ground_truth["images"]

    log_results = {}
    categories = ["Male prediction", "Young prediction", "Smile prediction"]
    for filename in os.listdir(eval_dir):
        if filename.lower().endswith((".jpg")):
            full_path = os.path.join(eval_dir, filename)
            print("Processing:", filename)
            arr = preprocess_image(full_path)

            if(is_keras):
                output, ms = run_keras_model(arr, model)
            else:
                output, ms = run_tflite_model(arr, model)

            log_results[filename] = dict(zip(categories, output))
            log_results[filename].update({"Time": ms})
            log_results[filename].update(images[filename]["labels"])

    return log_results

## Results of Test model performance
def output_results(results: Dict[str, Any], model_name: str =""):
    """
    Write evaluation results to a CSV file and optionally print a formatted table.

    Output file is`<model_name>_results.csv`. If debugging is enabled via
    `RUN_DEBUG`.

    Args:
        results: A dictionary mapping filenames to their evaluation results, including
            - predictions,
            - inference time
            - ground‑truth labels.
        model_name: Prefix for the output CSV filename.
    """
    target_file = model_name + "_results.csv"
    with open(target_file, "w", newline="") as f:
        fieldnames = ["Filename", "Male prediction", "Young prediction", "Smile prediction", "Time", "Male", "Young", "Smiling"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in results.items():
            v.update({"Filename": k})
            writer.writerow(v)

    if RUN_DEBUG:
        header =(
            f'{"Filename":<15} '
            f'{"Male prediction":<20} '
            f'{"Young prediction":<20} '
            f'{"Smile prediction":<20} '
            f'{"Time":<15}'
        )
        print(header)
        print("-" * len(header))

        for image, result in results.items():
            row = (f"{image:<15} {result["Male prediction"]:<20.15f} {result["Young prediction"]:<20.15f}"
                   f" {result["Smile prediction"]:<20.15f} {result["Time"]:<20.15f}")
            print(row)


def choose_model() -> str:
    """
    Prompt the user to select a model by number.

    Returns:
        str: The selected model filename.

    Raises:
        ValueError: If the user enters a number outside the valid range.
        """
    choice = int(input("Select a model by number: "))
    if 1 <= choice <= len(MODELS):
        # model index starts at 1
        return  MODELS[choice-1]
    else:
        raise ValueError("Invalid model selection")


if __name__ == "__main__":
    print("Available models:\n")
    for i, m in MODELS_MAPPING.items():
        print(f"{i}. {m}")

    model_name = choose_model()
    is_keras = model_name.rsplit(".")[1] == "keras"

    model = load_keras_model(model_name) if is_keras else load_tflite_model(model_name)
    print(f"Evaluating {model_name}")

    results = evaluate_model(model, is_keras)
    output_results(results, model_name=model_name)