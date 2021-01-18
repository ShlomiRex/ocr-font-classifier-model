# This file is the final code that will be scored in the course.
# If you want to see the images, remove comment from '#TODO:' lines.
import h5py
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tensorflow.python.keras.backend import dropout
import seaborn as sn
import pandas as pd
import math
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib


# Is using GPU?
print("CPU:")
print(tf.config.experimental.list_physical_devices('CPU'))
print("GPU:")
print(tf.config.experimental.list_physical_devices('GPU'))

# Constants
FONTS = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']
# Pre-calculated average width, height of all cropped train data
AVG_CHAR_WIDTH = 28
AVG_CHAR_HEIGHT = 49

train_filename = "train/SynthText.h5" # Original set
train_filename2 = "train/train.h5" # 18.1.2021 new training set
val_filename = "validation/SynthText_val.h5"



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
    font = db['data'][img_name].attrs['font']     # Contains list of fonts.
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
        word_font = font[char_index_accumulator].decode()
        chars = []

        word_bb = wordBB[:, :, word_index]
        word_crop = crop_affine(img, word_bb)

        # Process chars
        for char_index in range(len(word)):
            char = chr(word[char_index])
            char_font = font[char_index_accumulator].decode()
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
            "font": word_font,
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


# func: Plot sample (train, validation)
def plot_sample(x, y, index):
    """
    Plot sample by given index.
    """
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(y[index])


# func: Draw bounding box (for debugging purposes)
def draw_bb(bb):
    # Draw bb points
    formats__ = ["rp", "gp", "bp", "wp"]
    colors = ["r", "g", "b", "w"]
    x_line = []
    y_line = []
    for i in range(4):
        x = bb[0][i]
        y = bb[1][i]
        plt.plot(x, y, formats__[i])
        x_line.append(x)
        y_line.append(y)
    x_line.append(bb[0][0])
    y_line.append(bb[1][0])
    plt.plot(x_line, y_line, color="red")


# func: Plot training data with subplots (for debugging purposes)
def plot_samples(X, Y, low=0, high=28, fig_size=8):
    plt.figure(figsize=(fig_size, fig_size))
    j = 0

    row_col = round(math.sqrt(high-low))+1
    print(row_col)
    for i in range(low, high):
        plt.subplot(row_col, row_col, j+1)
        j += 1
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap="gray")
        plt.xlabel(Y[i])
    plt.show()


# func: Convert image to noisy image
def noisy(image, noise_type="random"):
    """
    Create noise in the image.
    noise_type is the noise type. Default is random.
    """
    row = image.shape[0]
    col = image.shape[1]

    if noise_type == "random":
        #noise_type = random.choice(["gauss", "s&p", "poisson", "speckle"])
        noise_type = random.choice(["gauss", "s&p", "poisson"])
        #noise_type = random.choice(["gauss", "poisson"])
        #noise_type = random.choice(["gauss"])

    if noise_type == "gauss":
        mean = 0
        var = 0.007
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


# func: Process image and label and append to set (training, validation)
# Ubuntu Mono = index 0
# Skylark = index 1
# Sweet Puppy = index 2
def append_to_set(X, Y, x, y, _noisy=True):
    """
    Append (x,y) sample to (X,Y) arrays. Checking correct font (y) and shape of image (x).
        X, Y - Images and labels set
        x,y - Image and label to append to set
    Set noisy to False if you don't want to convert image 'x' to noisy image and appending it (append 'x' without modification).
    """
    # Convert to gray
    try:
        if x.shape[2] != 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    except:
        pass
    # Resize
    if x.shape[0] != AVG_CHAR_HEIGHT or x.shape[1] != AVG_CHAR_WIDTH:
        x = cv2.resize(x, (AVG_CHAR_WIDTH, AVG_CHAR_HEIGHT))
    # Normalize
    x = normalize(x)

    # Map y string to float
    if type(y) == str:
        if y == "Ubuntu Mono":
            y = 0
        elif y == "Skylark":
            y = 1
        elif y == "Sweet Puppy":
            y = 2
        else:
            raise "Error font, no such font: " + str(y)

    if _noisy:
        x = noisy(x)

    X.append(x)
    Y.append(y)


# func: Create x,y sets from h5 file.
def populate(filename, X, Y, _noisy=True):
    """
    filename - h5 file to read from
    X - array to populate
    Y - array to populate
	_noisy - If you want to convert X set to noisy images (for training). If it's validation set, set this to False.
    """

    # Read from db
    db = h5py.File(filename, "r")
    im_names = list(db["data"].keys())
    num_of_images = len(im_names)
    print(f"Number of images: {num_of_images}")

    for img_name in im_names:
        res = extract_data(db, img_name)
        for word in res["words"]:
            for char in word["chars"]:
                char_font = char["font"]
                char_crop = char["crop"]

                # There are some images with defect bounding boxes (image: hubble_22.jpg)
                if char_crop.shape[0] == 0 or char_crop.shape[1] == 0:
                    word_str = word["word"]
                    char_str = char["char"]
                    print(
                        f"Invalid crop at image: {img_name}, word: {word_str}, char: {char_str}")
                else:
                    append_to_set(X, Y, char_crop, char_font, _noisy)


# func: Plot image
def plot_image(filename, index):
    db = h5py.File(filename, "r")
    im_names = list(db["data"].keys())
    img_name = im_names[index]

    img = db['data'][img_name][:]
    plt.figure()
    plt.imshow(img)

    wordBB = db['data'][img_name].attrs['wordBB']

    draw_bb(wordBB)

    res = extract_data(db, img_name)

    plt.figure()
    plt.imshow(res["words"][0]["crop"])

    word = res["words"][0]

    for char in word["chars"]:
        plt.figure()
        plt.imshow(char["crop"])


# func: Predict on cropped images
def get_predictions(model, X):
	"""
	model - model to use
	X - Set of images to predict font (after cropping! It's important!)
	"""
	predictions = model.predict(X)

	result = []

	for i in range(predictions.shape[0]):
		m = np.max(prediction[i]) # Maximum prediction (highest certainty)
		font_predicted = list(np.where(prediction[i] == m))[0][0]
		result.append(font_predicted)

	return result


########################################################################################################################################################
######################################################## Program starts here ###########################################################################
########################################################################################################################################################



# Training set (original training set)
x_train = []
y_train = []
populate(train_filename, x_train, y_train)


# Additional training set (18.1.2021)
populate(train_filename2, x_train, y_train)


print(f"x_train length: {len(x_train)} y_train length: {len(y_train)}")


#TODO: Show/hide
#print("Will now display images from training set...")
#plot_image(train_filename, 10) # plot original image
#plot_samples(x_train, y_train, low=520, high=550) # plot processed training images
#plt.show() 




# Validation set
db = h5py.File(val_filename, "r")
IM_NAMES = list(db["data"].keys())

num_of_images = len(IM_NAMES)
print(f"Validation - Number of images: {num_of_images}")

x_val = [] #Images
y_val = [] #Labels

populate(val_filename, x_val, y_val, _noisy=False) #Validation is without noise
print(f"x_val length: {len(x_val)} y_val length: {len(y_val)}")


#TODO: Show/hide
#print("Will now display images from validation set...")
#plot_image(val_filename, 519)
#plot_samples(x_val, y_val)
#plt.show() 





# Create model
input_shape = (AVG_CHAR_HEIGHT, AVG_CHAR_WIDTH, 1)
model = tf.keras.models.Sequential(
    [
        keras.Input(shape=input_shape),
        
        layers.Conv2D(128, kernel_size=(9, 9), activation="relu"),
        
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, kernel_size=(3, 3), activation="relu"),

        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
		layers.Dropout(0.1),
        layers.Dense(1024, activation='relu'),
		layers.Dropout(0.1),
        layers.Dense(3, activation="softmax"),
    ]
)

#OPTIMIZERS = ["SGD", "adam", "adadelta", "adagrad"]
#optimizer = keras.optimizers.Adam(learning_rate=.000085)
optimizer = keras.optimizers.Adamax(learning_rate=.00004)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.build()
model.summary()



# Train

# Convert python list to np array
X_train = np.array(x_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
Y_train = np.array(y_train)


X_val = np.array(x_val)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
Y_val = np.array(y_val)

print("X train shape: ", X_train.shape)
print("Y train shape: ", Y_train.shape)

print("X val shape: ", X_val.shape)
print("Y val shape: ", Y_val.shape)

epoch = 50 #TODO: Change to 50 in final version
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epoch) 


# Draw result for training

# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
try:
    plt.plot(history.history['val_accuracy'])
except:
    pass
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
try:
    plt.plot(history.history['val_loss'])
except:
    pass
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()




# Evaluate
print("Evaluate on train data")
results = model.evaluate(X_train, Y_train, batch_size=128)
print("train loss, train acc:", results)

print("Evaluate on validation data")
results = model.evaluate(X_val, Y_val, batch_size=128)
print("val loss, val acc:", results)







# Predict on validation set for confusion matrix
prediction_set_images = X_val 
prediction_set_labels = Y_val # ground truth

prediction = model.predict(prediction_set_images)

confusion_predictions = []

for i in range(prediction.shape[0]):
    predictions = prediction[i]
    m = np.max(prediction[i]) # Maximum prediction
    font_predicted = list(np.where(prediction[i] == m))[0][0]
    font_truth = prediction_set_labels[i]
    success = (font_predicted == font_truth)

    confusion_predictions.append([font_predicted, font_truth, m, success])







# Confusion matrix
array = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0 ,0]
]

# Rows = Predicted
# Cols = Ground truth

for x in confusion_predictions:
    # Select where to put in confusion matrix
    row = None
    col = None
    
    # Unwrap x
    font_predicted = x[0]
    font_truth = x[1]
    prediction_acc = x[2]
    success = x[3]
    
    # If predicted successfully
    if success:
        # Then put in (i, i)
        row = col = font_predicted
    
    # Wrong prediction
    else:
        col = font_truth
        row = font_predicted
    
    array[row][col] += 1


df_cm = pd.DataFrame(array, 
                     index = FONTS,
                     columns = FONTS)

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()






# Calculate accuracy from confusion matrix
yes = array[0][0] + array[1][1] + array[2][2] # when index i = j
no = array[0][1] + array[0][2] + array[1][0] + array[1][2] + array[2][0] + array[2][1] # when index i != j

total = yes+no

acc = yes/total
print("Accuracy: ", acc)






# Save model
model.save("saved_model.h5")




########################################################################################################################################################
######################################################## Training ends here ############################################################################
########################################################################################################################################################


# A customer will have the model (.h5 file) and this is the code he will need to execute for predictions or evaluation.

"""
# Load model
model = None
model = keras.models.load_model("saved_model.h5")
model.summary()

# Now customer can use this model to predict images.
# TODO: Create prediction function that gets raw images (3 channels, diffirent sizes) and crop them, and predict on them
#model.pred
#plot_samples(X_val, Y_predicted)
predictions = get_predictions(model, X_val[0:10])
print(predictions)

# Let's evaluate training and validation set to be sure it works
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_train, Y_train, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on validation data")
results = model.evaluate(X_val, Y_val, batch_size=128)
print("val loss, val acc:", results)
"""