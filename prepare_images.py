import os
import cv2
import numpy as np
import random


def prepare(root_directory):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(root_directory):
        path = root.split(os.sep)
        for file in files:
            image = cv2.imread(root + '\\' + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            image = np.reshape(image, image.shape + (1,))
            if 'train' in root:
                train_images.append(image)
                train_labels.append(int(root[-1]))
            elif 'eval' in root:
                test_images.append(image)
                test_labels.append(int(root[-1]))

    # shuffle
    train_labels = np.reshape(train_labels, np.asarray(train_labels).shape + (1,))
    train_temp = list(zip(train_images, train_labels))
    random.shuffle(train_temp)
    train_images, train_labels = zip(*train_temp)

    test_temp = list(zip(test_images, test_labels))
    random.shuffle(test_temp)
    test_images, test_labels = zip(*test_temp)

    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)


def prepare_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = np.reshape(image, image.shape + (1,))
    return image

