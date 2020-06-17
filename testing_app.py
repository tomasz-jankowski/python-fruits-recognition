# Import required libraries
import cv2
from os.path import os, dirname
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random

# List of categories (directories names)
CATEGORIES = ["bad_apple", "bad_grape", "bad_pear", "cherry", "good_apple", "good_avocado", "good_grape", "good_pear", "ripe_avocado"]

# Level 2 - display information about errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Commented line = use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Source folder path
main_dir = dirname(os.path.abspath(__file__))

# Paths to image database (train, test, all)
training_dir = os.path.join(main_dir, 'database', 'training')
testing_dir = os.path.join(main_dir, 'database', 'testing')
all_dir = os.path.join(main_dir, 'database', 'all')

# Currently used path
DATADIR = testing_dir

# Load all images and save them to array variable
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        break
    break

# Variable to store training data
testing_data = []


# Function that converts previously created data array to a test data array
def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                testing_data.append([img_array, class_num])
            except Exception as e:
                pass


# Call the function
create_testing_data()

# Shuffle test data
random.shuffle(testing_data)

# Create array variables to store objects and labels
X = []
y = []

# Save objects and labels to arrays
for features, label in testing_data:
    X.append(features)
    y.append(label)

# Convert arrays to NumPy matrices
X = np.array(X)
y = np.array(y)

# Change the value range from 0-255 to 0-1
X = X / 255.0

# Load the trained model from given path
keras_model_path = os.path.join(main_dir, 'models', 'test')
model = tf.keras.models.load_model(keras_model_path)

# Display model summary
model.summary()

# Display information about the effectiveness of test data classification
loss, acc = model.evaluate(X, y, verbose=2)
print('Accuracy: {:5.2f}%'.format(100 * acc))
print('Loss: {:5.2f}'.format(loss))