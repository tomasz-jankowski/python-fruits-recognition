# Import required libraries
from os.path import os, dirname
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tqdm import tqdm

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
DATADIR = training_dir

# List of categories (directories names)
CATEGORIES = ["bad_apple", "bad_grape", "bad_pear", "cherry", "good_apple", "good_avocado", "good_grape", "good_pear",
              "ripe_avocado"]

# Load all images and save them to array variable
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        break
    break

# Variable to store training data
training_data = []


# Function that converts previously created data array to a test data array
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass


# Call the function
create_training_data()

# Shuffle test data
random.shuffle(training_data)

# Create array variables to store objects and labels
X = []
y = []

# Save objects and labels to arrays
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert arrays to NumPy matrices
X = np.array(X)
y = np.array(y)

# Change the value range from 0-255 to 0-1
X = X / 255.0

# ------------------------------
# Neural network architecture
# ------------------------------

# Early Stopping - stop learning after 3 epochs without any improvement
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Sequential network declaration
model = models.Sequential()

# Dropout - layer killing random neurons (100x100, random seed: 3)
model.add(Dropout(0.1, input_shape=(100, 100, 3)))

# Layers - 3x Conv2D and 2x MaxPooling2D
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # input_shape=(100, 100, 3)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Input flattening (conversion from matrix to vector)
model.add(layers.Flatten())

# Dropout layer again - 10% chance for neuron to be killed
model.add(Dropout(0.1))

# Two closely connected layers of the neural network
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(9))

# Developed netowrk model compilation using the configuration below (Adam optimization algorithm)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train configured model
history = model.fit(X, y, validation_split=0.01, epochs=2, callbacks=[callback])

# Save the trained model to a given path
keras_model_path = os.path.join(main_dir, 'models', 'test')
model.save(keras_model_path)