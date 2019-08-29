#!/usr/bin/python3

import os
import cv2
import numpy as np
from PIL import Image

from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.feature import hog

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as matimg


def resize_image(input_path, output_path, size):
    original_image = Image.open(input_path)
    resized_image = original_image.resize(size)
    resized_image.show()
    resized_image.save(output_path)


def get_mse_ssi(original_image, listed_image, title):
    mse = measure.compare_mse(original_image, listed_image)
    ssi = measure.compare_ssim(original_image, listed_image)
    fig = plt.figure(title)
    plt.suptitle("Mean Square Error: %.2f, Structural Similarity Index: %.2f" % (mse, ssi))

    images = ("original_image", original_image), ("Image in saved List", listed_image)
    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

    plt.show()

    return mse, ssi


def detect_license_plate(license_plate_dimensions, gray_captured_frame, binary_captured_frame, label_image):

    min_height, max_height, min_width, max_width = license_plate_dimensions
    license_plate_image = []
    license_plate_geometry = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_captured_frame, cmap="gray")

    for i in regionprops(label_image):
        if i.area > 50:
            print("There is a license plate in the given image frames")
            min_row, min_col, max_row, max_col = i.bbox
            # print(min_row, min_col, max_row, max_col)
            height = max_row - min_row
            width = max_col - min_col
            if height >= min_height:
                if height <= max_height:
                    if width >= min_width:
                        if width <= max_width:
                            if width > height:
                                license_plate_image.append(binary_captured_frame[min_row:max_row,
                                                          min_col:max_col])
                                license_plate_geometry.append((min_row, min_col,
                                                                 max_row, max_col))
                                rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                                               edgecolor="blue", linewidth=4, fill=False)
                                ax1.add_patch(rectBorder)
        else:
            print("There is no license plate in the given image frames")
    return plt.show(), license_plate_image


def segment(lic_plate, plate_with_label, each_letter_dimensions):

    min_height, max_height, min_width, max_width = each_letter_dimensions
    letters = []
    letter_order = []
    fig, ax1 = plt.subplots(1)
    ax1.imshow(lic_plate, cmap="gray")

    for i in regionprops(plate_with_label):
        y0, x0, y1, x1 = i.bbox
        height = y1 - y0
        width = x1 - x0

        if height > min_height:
            if height < max_height:
                if width > min_width:
                    if width < max_width:
                        altered_data = lic_plate[y0:y1, x0:x1]
                        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="green",
                                                        linewidth=3, fill=False)
                        ax1.add_patch(rect_border)
                        altered_char = resize(altered_data, (20, 20))
                        letters.append(altered_char)
                        letter_order.append(x0)
        else:
            print("Segmentation cannot be done for this image with current settings")
    return plt.show(), letters, letter_order


def process_data_training(training_directory, characters):
    each_data = []
    valid_data = []
    for j in characters:
        for k in range(10):
            image_path = os.path.join(training_directory, j, j + '_' + str(k) + '.jpg')
            img_details = imread(image_path, as_gray=True)
            bin_image = img_details < threshold_otsu(img_details)
            flatten_image = bin_image.reshape(-1)
            each_data.append(flatten_image)
            valid_data.append(j)
    return np.array(each_data), np.array(valid_data)


def classify(model, letters):
    result = []
    for every in letters:
        every = every.reshape(1, -1)
        res = model.predict(every)
        result.append(res)
    return result


def predict(classified_result, letter_order):
    plate_string = ''
    for m in classified_result:
        plate_string += m[0]

    return plate_string


def get_hog_features(img, vis):
    if vis == True:
        features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  transform_sqrt=True, visualize=vis, feature_vector=True)
        return features, hog_image
    else:
        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       transform_sqrt=True, visualize=vis, feature_vector=True)
        return features


def extract_features(img, color_space='RGB'):
    features = []
    for file in img:
        image = matimg.imread(file)
        hog_features = []
        vis = False
        if color_space != 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], vis))
            hog_features = np.ravel(hog_features)
        else:
            feature_image = np.copy(image)
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], vis))
            hog_features = np.ravel(hog_features)

        features.append(hog_features)
    return features
