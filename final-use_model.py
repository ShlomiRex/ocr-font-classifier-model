
# This is the file that the customer will use to predict images.
# A customer will have the model (.h5 file) and this is the code he will need to execute for predictions or evaluation.

import h5py
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import csv
import os
import datetime


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


# func: Extract data from image name return json
# Note: this function is modified version from the training file. (doesn't get font)
def extract_data(db, img_name: str, font_check=True):
	"""
	Process the image and returned processed result.
	Parameter db is h5 database read from file.
	Set font_check to False to not include font (for test set)
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
	if font_check:
		font = db['data'][img_name].attrs['font']     # Contains list of fonts. # TODO: Remove in production
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
		# word_font = font[char_index_accumulator].decode()
		chars = []

		word_bb = wordBB[:, :, word_index]
		word_crop = crop_affine(img, word_bb)

		# Process chars
		for char_index in range(len(word)):
			char = chr(word[char_index])
			if font_check:
				char_font = font[char_index_accumulator].decode() # TODO: Remove for production
			else:
				char_font = None
			char_bb = charBB[:, :, char_index_accumulator]

			# assert char_font == word_font # Double check that the pre-processed image is indeed 1 font per word, and each char is same font as word.

			crop_char = crop_affine(img, char_bb)

			chars.append({
				"char": char,
				"font": char_font,
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


# func:
def populate_to_predict(filename, lst, validation=True):
	"""
	filename - h5 file to read from
	lst - python list to populate
	validation - Set to False . Will not check font (in test set there is no font)
	"""
	# Read from db
	db = h5py.File(filename, "r")
	im_names = list(db["data"].keys())
	for img_name in im_names:
		res = extract_data(db, img_name, font_check=validation)
		for word in res["words"]:
			for char in word["chars"]:
				char_crop = char["crop"] # image
				char_str = char["char"] # english character for the cropped image
				font = char["font"] #TODO: Remove for production
				# To gray
				char_crop = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)

				# Resize
				char_crop = cv2.resize(char_crop, (AVG_CHAR_WIDTH, AVG_CHAR_HEIGHT))


				# There are some images with defect bounding boxes (image: hubble_22.jpg)
				if char_crop.shape[0] == 0 or char_crop.shape[1] == 0:
					word_str = word["word"]
					print(f"Invalid crop at image: {img_name}, word: {word_str}, char: {char_str}")
				else:
					lst.append({
						"image_name": img_name,
						"char": char_str,
						"char_crop": char_crop
					})




# Load model

model = None
model = keras.models.load_model("saved_model.h5")
model.summary()


# Pre-calculated average width, height of all cropped train data
AVG_CHAR_WIDTH = 28
AVG_CHAR_HEIGHT = 49

#val_filename = "validation/SynthText_val.h5"
test_filename = "test/test.h5"





# predict
to_predict = []
print("Loading images...")
populate_to_predict(test_filename, to_predict, validation=False)
print("Number of images to predict: ", len(to_predict))

print("Converting to numpy array...")
images = np.array([x["char_crop"] for x in to_predict])
images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

now = datetime.datetime.now()
print("Predicting now...")
p = model.predict_classes(images)
now2 = datetime.datetime.now()
delta = (now2-now)
print(f"Prediction time (seconds): {delta.seconds}")

print("Writing to csv...")
rows = []
outfile = "out.csv"
with open(outfile, "w", newline='') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',')
	# Write header
	csvwriter.writerow(["", "image", "char", "Skylark", "Sweet Puppy", "Ubuntu Mono"])

	# Count
	c_sky = 0
	c_sweet = 0
	c_ubuntu = 0

	for i in range(len(to_predict)):
		x = to_predict[i]
		font = p[i]
		x["font_predicted"] = font
		image_name = x["image_name"]
		char = x["char"]
		#char_crop = x["char_crop"]
		#print(f"Image: {image_name} Char: {char} Font: {font}")
		fonts = [0, 0, 0] # in csv order (first index = sky, second index = sweet, third index = ubuntu)
		# In my model:
		# Ubuntu Mono = index 0
		# Skylark = index 1
		# Sweet Puppy = index 2

		if font == 0:
			# Ubuntu Mono
			fonts[2] = 1
			c_ubuntu += 1
		elif font == 1:
			# Skylark
			fonts[0] = 1
			c_sky += 1
		else:
			# Sweet Puppy
			fonts[1] = 1
			c_sweet += 1

		row = [str(i), image_name, char, str(fonts[0]), str(fonts[1]), str(fonts[2])]
		rows.append(row)
		csvwriter.writerow(row)

print("CSV complete")
print(f"# Skylark: {c_sky}")
print(f"# Sweet Puppy: {c_sweet}")
print(f"# Ubuntu Mono: {c_ubuntu}")
print(f"Total (sum): {c_sky+c_sweet+c_ubuntu}")

print("CSV output is located at: " + os.path.abspath(outfile))

print("Thanks for using my model :) I hope you have a great day!")












"""
# check result

with open("validation/char_font_preds.csv", "r") as predicion_csv_validation:
	reader = csv.reader(predicion_csv_validation, delimiter=',')
	next(reader) # Skip first line
	i = 0
	c_sky = 0
	c_sweet = 0
	c_ubuntu = 0
	for row in reader:
		img_name = row[1]
		char = row[2]
		skylark = int(float(row[3]))
		sweetpuppy = int(float(row[4]))
		ubuntumono = int(float(row[5]))

		if skylark == 1:
			c_sky += 1
		elif sweetpuppy == 1:
			c_sweet += 1
		else:
			c_ubuntu += 1

		i += 1

print(f"Total : {c_sky} {c_sweet} {c_ubuntu} ")
"""
