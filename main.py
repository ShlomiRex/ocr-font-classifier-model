import h5py
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

FONTS = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']

file_name = "font_recognition_train_set/SynthText.h5"

db = h5py.File(file_name, "r")
im_names = list(db["data"].keys())

for i in range(len(im_names)):
    img_name = im_names[i]
    img = db['data'][img_name][:]
    font = db['data'][img_name].attrs['font']
    txt = db['data'][img_name].attrs['txt']
    charBB = db['data'][img_name].attrs['charBB']
    wordBB = db['data'][img_name].attrs['wordBB']

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    print(f"Image: {img_name}, Text: {txt}, Font: {font}")

    cv2.imshow('dst', img)
    """

    edges = cv.Canny(img, 100, 200)

    nC = charBB.shape[-1]
    plt.figure()
    plt.imshow(img)
    for b_inx in range(nC):
        if (font[b_inx].decode('UTF-8') == FONTS[0]):
            color = 'r'
        elif (font[b_inx].decode('UTF-8') == FONTS[1]):
            color = 'b'
        else:
            color = 'g'
        bb = charBB[:, :, b_inx]
        x = np.append(bb[0, :], bb[0, 0])
        y = np.append(bb[1, :], bb[1, 0])
        plt.plot(x, y, color)
    nW = wordBB.shape[-1]
    for b_inx in range(nW):
        bb = wordBB[:, :, b_inx]
        x = np.append(bb[0, :], bb[0, 0])
        y = np.append(bb[1, :], bb[1, 0])
        plt.plot(x, y, 'k')

    plt.imshow(img)
    plt.show()

    plt.imshow(edges, cmap='gray')
    plt.show()


"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py

file_name = 'font_recognition_train_set/SynthText.h5'
db = h5py.File(file_name, 'r')
im_names = list(db['data'].keys())
im = im_names[110]
imgs = db['data'][im][:]
font = db['data'][im].attrs['font']
txt = db['data'][im].attrs['txt']
charBB = db['data'][im].attrs['charBB']
wordBB = db['data'][im].attrs['wordBB']

font_name = ['Skylark', 'Ubuntu Mono', 'Sweet Puppy']

nC = charBB.shape[-1]
plt.figure()
plt.imshow(imgs)
for b_inx in range(nC):
    if(font[b_inx].decode('UTF-8')==font_name[0]):
        color = 'r'
    elif(font[b_inx].decode('UTF-8')==font_name[1]):
        color = 'b'
    else:
        color = 'g'
    bb = charBB[:,:,b_inx]
    x = np.append(bb[0,:], bb[0,0])
    y = np.append(bb[1,:], bb[1,0])
    plt.plot(x, y, color)
nW = wordBB.shape[-1]
for b_inx in range(nW):
    bb = wordBB[:,:,b_inx]
    x = np.append(bb[0,:], bb[0,0])
    y = np.append(bb[1,:], bb[1,0])
    plt.plot(x, y, 'k')
plt.show()


"""