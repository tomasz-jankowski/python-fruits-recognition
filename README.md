# python-fruits-recognition
Deep learning university project to recognize fruits and their quality

[![Generic badge](https://img.shields.io/badge/python-3.7.7-blue.svg)](https://shields.io/)   [![Generic badge](https://img.shields.io/badge/anaconda-2019.10-green.svg)](https://shields.io/)   [![Generic badge](https://img.shields.io/badge/tensorflow-2.1.0-red.svg)](https://shields.io/)
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

## Usage

### Choose between CPU and GPU calculation
Below line of code lets you choose used processing unit, comment it to use GPU.
```
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### training_app.py
Artificial neural network architecture and training process.

### testing_app.py
Script used to verify neural network effectiveness on the test dataset.

### classify.py
Main script used to show neural network predictions on given images. Image should be given as path, without any quotation marks and apostrophes.

![classify](https://user-images.githubusercontent.com/48838669/84918313-bb408a80-b0c0-11ea-921d-22909eec8dbc.png)

## Anaconda packages being used
* numpy 1.18.1
* matplotlib 3.1.3
* opencv 3.4.2
* tensorflow 2.1.0
* tqdm 4.46.0
* tabulate 0.8.3