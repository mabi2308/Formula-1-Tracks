import tensorflow as tf
import numpy as np
import sys
import os

print("Loading model...")
model_path = "f1_track_classifier.keras"  # Keras format
model = tf.keras.models.load_model(model_path)
print(f"Model loaded from {model_path}")

dataset_dir = "dataset"
class_names = sorted(os.listdir(dataset_dir))
print("Classes found:", class_names)

def predict(image_path):
    print(f"Predicting image: {image_path}")
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    pred_index = np.argmax(preds)
    confidence = preds[0][pred_index]

    predicted_class = class_names[pred_index]
    print(f"Prediction: {predicted_class} ({confidence:.2%} confidence)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_track.py path_to_image")
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(f"File {img_path} does not exist.")
        sys.exit(1)

    predict(img_path)
