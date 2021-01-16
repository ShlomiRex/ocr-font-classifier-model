
# This is the file that the customer will use to predict images.
# A customer will have the model (.h5 file) and this is the code he will need to execute for predictions or evaluation.

import h5py
from matplotlib import pyplot as plt
import cv2
import numpy as np
#import seaborn as sn
import pandas as pd
import math
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib


# Re-use functions

# func: Crop the word with perfect angle - affine transformation
# Tutorial that helped me:
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
def crop_affine(img, bb):
    """
    Crop image using affine transformation, around bounding box. Returns cropped image.
    """
    img_copy = img.copy()
    width = img_copy.shape[1]
    height = img_copy.shape[0]

    point1 = (bb[0][0], bb[1][0])  # Top-left
    point2 = (bb[0][1], bb[1][1])  # Top-right
    point3 = (bb[0][2], bb[1][2])  # Bottom-Right
    point4 = (bb[0][3], bb[1][3])  # Bottom-Left

    # Euclidian distance
    bb_width = int(np.linalg.norm(np.array(point1) - np.array(point2)))
    bb_height = int(np.linalg.norm(np.array(point1) - np.array(point3)))

    # Mapping srcPoints (list of points of size 3) to dstPoints (list of points of size 3)
    srcTri = np.array([point1, point2, point4]).astype(np.float32)
    dstTri = np.array([[0, 0], [bb_width, 0], [0, bb_height]]
                      ).astype(np.float32)

    # Apply transformation
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    warp_dst = cv2.warpAffine(img_copy, warp_mat, (width, height))

    # Crop the 'warped' image
    crop = warp_dst[0:bb_height, 0:bb_width]

    return crop


# func: Normalize function
def normalize(img, low=0, high=1):
    """
    Normalize image to range [low, high] from any range. Note: fast algorithm.
    """
    return np.interp(img, [np.min(img), np.max(img)], [low, high])


# func: Extract data from image name return json
# Note: this function is modified version from the training file. (doesn't get font)
def extract_data(db, img_name: str):
    """
    Process the image and returned processed result.
    Parameter db is h5 database read from file.
    Return a json in the following structure (as an example):
    {
        "img": <ndarray>,
        "name": "test.png",
        "words": [
            {
                "word": "the",
                "font": "Ubuntu Mono",
                "chars": [
                    {
                        "char": "t",
                        "font": "Ubuntu Mono",
                        "crop": <ndarray>,
                        "bb": <ndarray>
                    }, ...
                ],
                "bb": <ndarray>
                "crop": <ndarray>
            }, ...
        ]
    }
    """
    img = db['data'][img_name][:]                 # The image.
    #font = db['data'][img_name].attrs['font']     # Contains list of fonts.
    txt = db['data'][img_name].attrs['txt']       # Contains list of words.
    # Contains list of bb for words.
    charBB = db['data'][img_name].attrs['charBB']
    # Contain list of bb for chars.
    wordBB = db['data'][img_name].attrs['wordBB']

    words = []
    char_index_accumulator = 0
    word_index = 0  # Counter

    # Process word
    for word in txt:
        # Convert bytes to string
        #word_font = font[char_index_accumulator].decode()
        chars = []

        word_bb = wordBB[:, :, word_index]
        word_crop = crop_affine(img, word_bb)

        # Process chars
        for char_index in range(len(word)):
            char = chr(word[char_index])
            #char_font = font[char_index_accumulator].decode()
            char_bb = charBB[:, :, char_index_accumulator]

            # assert char_font == word_font # Double check that the pre-processed image is indeed 1 font per word, and each char is same font as word.

            crop_char = crop_affine(img, char_bb)

            chars.append({
                "char": char,
                "font": None,
                "crop": crop_char,
                "bb": char_bb
            })

            char_index_accumulator += 1

        words.append({
            "word": word.decode(),
            "font": None,
            "chars": chars,
            "bb": word_bb,
            "crop": word_crop,
        })
        word_index += 1

    # Return result
    return {
        "img": img,
        "name": img_name,
        "words": words,
    }


# func: Predict fonts from raw database (images, and bounding boxes)
def predict_raw_h5_set(h5_path):
	"""
	A customer will use this function for each set of images he wants to predict.
	h5_path - h5 database path (images, and bounding boxes)
	"""
	# Read from db
	db = h5py.File(h5_path, "r")
	im_names = list(db["data"].keys())

	num_of_images = len(im_names)
	print(f"Number of images in set: {num_of_images}")
	images_for_prediction = []
	
	for img_name in im_names:
		json = extract_data(db, img_name)
		for word in json["words"]:
			for char in word["chars"]:
				crop = char["crop"]
				images_for_prediction.append(crop)

	print(f"Number of images for prediction: {len(images_for_prediction)}")
	
		


# Load model
model = None
model = keras.models.load_model("saved_model.h5")
model.summary()

# Now customer can use this model to predict images.
# TODO: Create prediction function that gets raw images (3 channels, diffirent sizes) and crop them, and predict on them
# model.pred
# plot_samples(X_val, Y_predicted)
predict_raw_h5_set("validation_set/SynthText_val.h5")
# predictions = get_predictions(model, X_val[0:10])
# print(predictions)
