import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_train.reshape((-1, 28, 28, 1))

# Define model architecture
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)),     # Input Layer
                                    tf.keras.layers.Dense(128, activation = 'relu'),    # Hidden Layer
                                    tf.keras.layers.Dense(10, activation = 'softmax')]) # Output Layer

# Compile Model & Train Model
model.compile(optimizier = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test))

# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open ('mode.tflite', 'wb') as f:
    f.write(tflite_model)
