import cv2
import os
import numpy as np
import pickle

def load_dataset(dataset_path):
    images = []
    labels = []
    label_dict = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for label_name in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_folder):
            continue

        if label_name not in label_dict:
            label_id = len(label_dict)
            label_dict[label_name] = label_id

        for image_name in os.listdir(label_folder):
            if not any(image_name.lower().endswith(ext) for ext in valid_extensions):
                continue

            image_path = os.path.join(label_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Image failed to load at path {image_path}")
                continue

            images.append(image)
            labels.append(label_id)

    return images, labels, label_dict

dataset_path = "/Users/apple/NEWFOLDER/Face-Detection-with-Name-Recognition-main/data"
images, labels, label_dict = load_dataset(dataset_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save("/Users/apple/NEWFOLDER/Face-Detection-with-Name-Recognition-main/trainer/trainer.yml")

with open('/Users/apple/NEWFOLDER/Face-Detection-with-Name-Recognition-main/label_dict.txt', 'wb') as f:
    pickle.dump(label_dict, f)
