# Import required libraries
import cv2
from os.path import os, dirname
import tensorflow as tf
import numpy as np
from tabulate import tabulate

# Assign an input argument to the variable
path = input('Podaj sciezke do obrazu: ')

# Level 2 - display information about errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Commented line = use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Source folder path
main_dir = dirname(os.path.abspath(__file__))

# Load the trained model from given path
keras_model_path = os.path.join(main_dir, 'models', '10epochs_dropout01-01_valsplit001__1')
model = tf.keras.models.load_model(keras_model_path)

# Save the image as NumPy matrix
test_image = cv2.imread(path)

# Increase matrix dimensions to match the requirements of the neural network
test_image = np.expand_dims(test_image, axis=0)

# Obtaining image prediction and converting the output to a list (table)
result = model.predict(test_image)
result = result[0].tolist()

# Adjusting the results to the range [0-1]
result[:] = [i / 10000 for i in result]
result[:] = [round(i, 4) for i in result]

# First column of the table
labels = ['Jablko (nieswieze)', 'Winogrono (nieswieze)', 'Gruszka (nieswieza)', 'Wisnia', 'Jablko (swieze)', 'Awokado (dojrzale)', 'Winogrono (swieze)', 'Gruszka (swieza)', 'Awokado (niedojrzale)']

# Print results along with the best prediction
print()
print(tabulate([[labels[4], result[4]],
                [labels[0], result[0]],
                [labels[7], result[7]],
                [labels[2], result[2]],
                [labels[5], result[5]],
                [labels[8], result[8]],
                [labels[6], result[6]],
                [labels[1], result[1]],
                [labels[3], result[3]]],
                headers=['Klasyfikacja', 'Prawdopodobienstwo'], tablefmt='orgtbl'))
print("\n Najlepsze dopasowanie: ", labels[result.index(max(result))] , " - pewnosc: ", max(result))