#!/usr/bin/python3

import cv2
import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


from all_function_call import resize_image
from all_function_call import get_mse_ssi
from all_function_call import detect_license_plate
from all_function_call import segment
from all_function_call import process_data_training
from all_function_call import classify
from all_function_call import predict
from all_function_call import get_hog_features
from all_function_call import extract_features

'''
# Here we give the image of the car while returning

'''

test_path = "./dataset/test_image/01_new.png"
# test_path = "./dataset/car_damage/WithDamage/h_Before.png"

'''
#  Here we store the images of the car taken while renting

'''

dataset_good_path = './dataset/car_damage_mean_ssi/WithoutDamage/*.png'

'''
# Here we store the data used for detecting the license plate of the car

'''

input_car_image = './dataset/test_image/my_image2.png'
#input_car_image = './dataset/test_image/my_image3.png'


'''
# Data used for extracting the features of the damged car and good car 

'''

car_with_damage = sorted(glob.glob('./dataset/car_damage/WithDamage/*.png', recursive=True))
# print("car_with_damage", car_with_damage)
car_without_damage = sorted(glob.glob('./dataset/car_damage/WithoutDamage/*.png', recursive=True))
# print("car_without_damage", car_without_damage)


'''
Make sure the size of the image of the car while returning matches with the size of the image of the car 
we took and saved them to compare while renting the car

'''

my_actual_image = cv2.imread(test_path)
im1 = Image.open(test_path)
width1, height1 = im1.size
print("Test image1 and image2 (which is images in the dataset) should be of same size in order to be compared:")
print("Printing the size of the input test image:")
print("Width and height of input image = ", width1, height1)

dataset_image_size_path = "./dataset/car_damage_mean_ssi/WithDamage/c_Before.png"
im = Image.open(dataset_image_size_path)
width, height = im.size
print("Printing the size of the image stored in the dataset:")
print("Width and height of dataset image = ", width, height)


if width1 == width and height1 == height:
    actual = my_actual_image
else:
    resize_image(input_path=test_path, output_path=test_path, size=(width, height))
    actual = cv2.imread(test_path)


'''
Image quality assessment of image given while returning
'''
actual = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)
actual_mse, actual_ssi = get_mse_ssi(actual, actual, "Original vs. Original")
print("Actual Mean square error for comparing original image with itself = ", actual_mse)
print("Actual Structural Similarity Index for comparing original image with itself = " + str(actual_ssi) + "\n")


'''
Comparing the MSE and SSIM with saved images in the dataset
'''

saved_image_list = glob.glob(dataset_good_path, recursive=True)
image_index = -1
mse = {}
ssid = {}
for idx, path in enumerate(saved_image_list):
    image_index += 1
    list_image = cv2.imread(saved_image_list[idx])
    list_image = cv2.cvtColor(list_image, cv2.COLOR_BGR2GRAY)
    Mean_square_error, structural_similarity_index = get_mse_ssi(actual, list_image,
                                                                 "Actual Image vs Images in the Dataset")
    print("Obtained Mean square error for comparing original image with listed image"
          + str(image_index) + " = " + str(Mean_square_error))
    print("Obtained Structural Similarity Index for comparing original image with listed image"
          + str(image_index) + " = " + str(structural_similarity_index) + "\n")
    mse[image_index] = Mean_square_error
    ssid[image_index] = structural_similarity_index


'''
Plotting the MSE comparision with saved images in the dataset
'''

mse_list = sorted(mse.items())
x, y = zip(*mse_list)
fig = plt.figure('Low mean square error means possible matching image')
plt.plot(x, y)
fig.suptitle('Mean Square error vs possible image match')
plt.xlabel('dataset image list index')
plt.ylabel('Mean Square Error')
plt.show()


'''
Plotting the SSIM comparision with saved images in the dataset
'''

ssid_list = sorted(ssid.items())
x, y = zip(*ssid_list)
fig = plt.figure('High structural_similarity_index means possible matching image')
plt.plot(x, y)
fig.suptitle('structural_similarity_index vs possible image match')
plt.xlabel('dataset image list index')
plt.ylabel('structural_similarity_index')
plt.show()


print("Image which has least Mean Square error and high structural_similarity_index would be a possible matching image")


"""

Scripts below are for identifying the license plate of the car

"""
captured_frame = imread(input_car_image, as_gray=True)

gray_captured_frame = captured_frame * 255

threshold_value = threshold_otsu(gray_captured_frame)
binary_captured_frame = gray_captured_frame > threshold_value

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_captured_frame, cmap="gray")
ax2.imshow(binary_captured_frame, cmap="gray")
plt.show()

label_image = measure.label(binary_captured_frame)
print("label_image = ", label_image.shape[0])


'''
Here we are trying to manually train the machine with user given dimensions of the License plate, which is not 
effective if the license plate of the car has different dimensions, Need to find a better solution in future

'''


# Below dimension works for my_image2.png
license_plate_dimensions = (0.08*label_image.shape[0], 0.2*label_image.shape[0],
                            0.15*label_image.shape[1], 0.4*label_image.shape[1])


# Below dimension works for my_image3.png but not as expected, need more analysis
# license_plate_dimensions = (0.05*label_image.shape[0], 0.08*label_image.shape[0],
#                             0.15*label_image.shape[1], 0.3*label_image.shape[1])

'''
Using the extracted dimensions of the car number plate and trying to draw a rectangle for the detected plate

'''
print("license_plate_dimensions = ", license_plate_dimensions)
fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_captured_frame, cmap="gray")
my_license_plate_plot, license_plate_image = detect_license_plate(license_plate_dimensions, gray_captured_frame,
                                                                  binary_captured_frame, label_image)


'''
Using the detected car number plate rectangle to extract each letters in the plate

'''

print("license_plate_image = ", license_plate_image)
lic_plate = np.invert(license_plate_image[0])
plate_with_label = measure.label(lic_plate)
print("", )
each_letter_dimensions = (0.35 * lic_plate.shape[0], 0.60 * lic_plate.shape[0],
                          0.05 * lic_plate.shape[1], 0.15 * lic_plate.shape[1])
fig, ax1 = plt.subplots(1)
ax1.imshow(lic_plate, cmap="gray")
my_segmented_plot, letters, letter_order = segment(lic_plate, plate_with_label, each_letter_dimensions)


'''
Train the machine with letters and Numbers for detecting the characters in the plates
'''

license_plate_train_path = './dataset/license_plate/train/'
characters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
each_data, valid_data = process_data_training(license_plate_train_path, characters)

'''
Predict the accuracy using the SVC model
'''

# Trying to see results with various kernals
svc_model_poly = SVC(kernel='poly', probability=True, gamma='auto')
accuracy_result_poly = cross_val_score(svc_model_poly, each_data, valid_data, cv=4)
print("Accuracy with poly kernel and 4 folds = ", accuracy_result_poly)


svc_model_rbf = SVC(kernel='rbf', probability=True, gamma='auto')
accuracy_result_rbf = cross_val_score(svc_model_rbf, each_data, valid_data, cv=4)
print("Accuracy with rbf kernel and 4 folds = ", accuracy_result_rbf)


svc_model = SVC(kernel='linear', probability=True, gamma='auto')
accuracy_result = cross_val_score(svc_model, each_data, valid_data, cv=4)
print("Accuracy with linear kernel and 4 folds = ", accuracy_result)

svc_model.fit(each_data, valid_data)


filename = './trained_model.sav'
pickle.dump(svc_model, open(filename, 'wb'))


model = pickle.load(open(filename, 'rb'))
classified_result = classify(model, letters)
print("Classified array list = ", np.array(classified_result))


predict_result = predict(classified_result, letter_order)
print('Predicted license plate letters = ', predict_result)

"""

Scripts below are used to extract the HOG features of Good car image and Damaged car image and divided 
the available data into test and train data to determine the classification accuracy using SVM and
K-Nearest Neighbour Classifier for understanding method works

"""

print("Total number of damaged car images: " + str(len(car_with_damage)))
print("Total number of good car images: " + str(len(car_without_damage)))

damage_img = cv2.imread(car_with_damage[7])
good_img = cv2.imread(car_without_damage[7])

figure, (damage_plot, good_plot) = plt.subplots(1, 2, figsize=(8, 4))

damage_plot.set_title('Damaged car image')
damage_plot.imshow(cv2.cvtColor(damage_img, cv2.COLOR_BGR2RGB))

good_plot.set_title('Good car image')
good_plot.imshow(cv2.cvtColor(good_img, cv2.COLOR_BGR2RGB))


# Extracting the HOG from features
ycrcb_damage_img = cv2.cvtColor(damage_img, cv2.COLOR_RGB2YCrCb)
ycrcb_good_img = cv2.cvtColor(good_img, cv2.COLOR_RGB2YCrCb)

visual = True
damage_features, damage_hog_image = get_hog_features(ycrcb_damage_img[:, :, 0], visual)
good_features, good_hog_image = get_hog_features(ycrcb_good_img[:, :, 0], visual)

figure, (damage_car_hog_plot, good_car_hog_plot) = plt.subplots(1, 2, figsize=(8, 4))

damage_car_hog_plot.set_title('Damage Car HOG feature')
damage_car_hog_plot.imshow(damage_hog_image, cmap='gray')

good_car_hog_plot.set_title('Good car HOG feature')
good_car_hog_plot.imshow(good_hog_image, cmap='gray')

plt.show()

# Training the model
t1 = time.time()

color_space = 'YCrCb'
damage_car_hog_features = extract_features(car_with_damage, color_space)
# print("damage_car_hog_features = ", damage_car_hog_features)
good_car_hog_features = extract_features(car_without_damage, color_space)
# print("good_car_hog_features = ", good_car_hog_features)


t2 = time.time()
print(round(t2-t1, 2), 'Seconds to extract HOG features...')

# array stack of feature vectors
X = np.vstack((damage_car_hog_features, good_car_hog_features)).astype(np.float64)

# Fit
X_scalar = StandardScaler().fit(X)

# Apply the scalar to X
scaled_X = X_scalar.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(damage_car_hog_features)), np.zeros(len(good_car_hog_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# print('Using: 9 orientations, (8, 8) pixels per cell and (2, 2) cell_per_block')
print('Size of the Training data set: ', len(X_train))
print('Size of the Testing data set: ', len(X_test))


# Trying to see the classification accuracy using the Linear SVC
print("Classification accuracy using the Linear Support Vector Machine Classifier: ")

# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2-t, 2), 'Time in sec to train Support Vector Machine Classifier...')

# Check the score of the SVC
print('Test score of my Support Vector Machine Classifier = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t = time.time()
n_predict = 50
print('Prediction Value using Support Vector Machine Classifier: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])

t2 = time.time()
print(round(t2-t, 5), 'Time in Sec to predict', n_predict, 'labels with Support Vector Machine Classifier')


# Trying to see the classification accuracy using the K-Nearest Neighbour Classifier
print("Classification accuracy using the K-Nearest Neighbour Classifier: ")

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
knn_pred = knn.predict(X_test)

# evaluate accuracy
print("K-Nearest Neighbour Classifier accuracy: {}".format(accuracy_score(y_test, knn_pred)))


# creating odd list of K for KNN
neighbors = list(range(1, 20))

# empty list that will hold cv scores
cv_scores = []
cv_scores_final = []

# perform 10-fold cross validation
print(" Performing 10 fold cross Validation, Hence cv = 10 is used ")
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
cv_scores_final.append(scores.mean())
print("Accuracy after 10 fold cross validation= ", cv_scores)
print("Accuracy after 10 fold cross validation= ", cv_scores_final)


# changing to mis classification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot mis classification error vs k
plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Mis classification Error")
plt.show()

'''
Note: This is not complete. As we see the accuracy from the extracted features is not good.
when we use image as a data to the machine, we need to realize the significance of image processing and dataset.
Since I could only collect very few images of car before and after damage, the analysis I could do is limited.

We need to consider two significant parameters such as,
1. SSIM between the images taken before renting and while returning, 
   ie., park the car in the same exact way and location, this is possible if we rent and return in same place
2. We need lot more image dataset of car before damage and after damage for training the machine
   
'''
