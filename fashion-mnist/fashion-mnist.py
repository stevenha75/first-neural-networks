import tensorflow as tf
from tensorflow import keras
# Keras is a high level API that simplifies the process of making a neural network
import numpy as np
import matplotlib.pyplot as plt

# Loading in data fashion_mnist data set
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shrinking down the data to make it easier to work w/
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    # Flattening the list so the neurons can take in the information
    # Necessary when taking in information from a 2d/2d array
    keras.layers.Flatten(input_shape=(28,28)), # input layer
    # Creating a dense layer (a fully connected layer; each neuron is connected to every other neuron in the network)
    # In our case, we will be using 128 neurons- this is usually based on 15-20% of the initial inputs
    # The activation function is a mathematical function that introduces non-linearity 
    # into the output of a neuron, allowing the neural network to model complex relationships between 
    # the input and output data.
    keras.layers.Dense(128, activation="relu"),
    # We choose relu (rectified linear unit) b/c it is fast and versatile
    keras.layers.Dense(10, activation="softmax") # output layer (10 neurons for the 10 classes; soft-max
    # was choosen b/c it allows us to get probability outputs)
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=7)

# gets the largest probability value and returns the index
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])    
    plt.show()
    # prediction[0] prints the probabilities for each neuron of the output layer
    # arg max returns the index of the largest number.