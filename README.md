# First-neural-networks
 A few of my first ML projects

## Table of Contents
- [Fashion MNIST](https://github.com/stevenha75/first-neural-networks/edit/main/README.md#fashion-mnist)
- [Installation & Usage]()
  - [Dependencies]()
  - [Installation]()
  - [Usage]()

# Fashion MNIST Neural Network Overview
This code demonstrates a neural network that classifies images of clothing items in the Fashion MNIST dataset. The code is structured to be easy to understand, with necessary packages imported and the dataset loaded in. The pixel values of the images are then normalized for ease of use by the neural network. The neural network is created using the keras.Sequential() method, which defines a flattened input layer, a hidden layer with 128 neurons, and an output layer with 10 neurons. The output layer is created with 10 neurons to represent the 10 different possible class/item types. The output layer returns the probabilities of each class type, which are then used to choose the largest value in order to classify the image type using the neural network. The model is compiled with the adam optimizer, sparse_categorical_crossentropy loss function, and accuracy metric. The model is then trained with the training data and evaluated with the testing data. The predict() method is used to obtain predictions on the test data, and the results are displayed for the first 5 test images using Matplotlib, providing a clear visualization of the model's performance. 

# Installation & Usage
- As of right now this project is not ready for public usage.
## Dependencies

## Installation
First, download the files onto your computer
```shell
git clone https://github.com/stevenha75/first-neural-networks.git
cd first-neural-networks
```
Then, install the needed packages for the neural network
```shell
```
## Using the neural networks
