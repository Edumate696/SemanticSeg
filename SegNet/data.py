import os
import pathlib
import sys

import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm


class DataConfig:
    # Set some parameters
    im_width = 128
    im_height = 128
    border = 5


class DataLoader:
    def __init__(self):
        base_path = pathlib.Path(__file__).parent.parent / 'Dataset' / 'train'

        ids = next(os.walk(base_path / "images"))[2]  # list of names all images in the given path
        print("[INFO] No. of images found = ", len(ids))
        sys.stdout.flush()

        X = np.zeros((len(ids), DataConfig.im_height, DataConfig.im_width, 1), dtype=np.float32)
        y = np.zeros((len(ids), DataConfig.im_height, DataConfig.im_width, 1), dtype=np.float32)

        # tqdm is used to display the progress bar
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            # Load images
            img = load_img(base_path / "images" / id_, color_mode="grayscale")
            x_img = img_to_array(img)
            x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
            # Load masks
            mask = img_to_array(load_img(base_path / "masks" / id_, color_mode="grayscale"))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
            # Save images
            X[n] = x_img / 255.0
            y[n] = mask / 255.0

        # Split train and valid
        self.__data = train_test_split(X, y, test_size=0.1, random_state=42)

    def load_dataset(self):
        X_train, X_valid, y_train, y_valid = self.__data
        print("[INFO] No. of training samples =", len(X_train))
        print("[INFO] No. of validation samples =", len(X_valid))
        return self.__data
