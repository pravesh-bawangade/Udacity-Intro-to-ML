import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import argparse
import json
from PIL import Image


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image


def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)

    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))

    top_values, top_indices = tf.math.top_k(prediction, top_k)
    print("These are the top propabilities", top_values.numpy()[0])

    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Udacity Image Classifier")
    parser.add_argument('--model_path', nargs='?', default='./1591029114.h5',
                        help='Model Path')
    parser.add_argument('--img_path', nargs='?', default='./test_images/orange_dahlia.jpg',
                        help='Image Location')
    parser.add_argument('--top_k', nargs='?', default=5,
                        help='Return the top K most likely classes')
    parser.add_argument('--category_names', nargs='?', default='./label_map.json',
                        help='Path to a JSON file mapping labels to flower names:')
    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    top_value, top_classes = predict(args.img_path, model, int(args.top_k))


