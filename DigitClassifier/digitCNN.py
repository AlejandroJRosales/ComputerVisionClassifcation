from keras.models import Sequential
from keras.layers import Reshape, Conv2D, MaxPool2D, Flatten, Dense
import pandas as pd
import numpy as np
# from hyperdash import Experiment

training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")

testing_images = np.array(testing_data)

(train_images, train_labels) = (np.array(training_data.iloc[:21000, 1:]), np.array(training_data.iloc[:21000, 0]))
(test_images, test_labels) = (np.array(training_data.iloc[21000:, 1:]), np.array(training_data.iloc[21000:, 0]))
(train_images, test_images) = (train_images/255.0, test_images/255.0)

# exp = Experiment("Digit Classifier")
# accuracy = exp.param("accuracy", 1.0)

model = Sequential()
model.add(Reshape(input_shape=(784,), target_shape=(28, 28, 1)))
model.add(Conv2D(kernel_size=(7, 7), filters=32, activation="relu"))
model.add(MaxPool2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, batch_size=64, epochs=25, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy} \t Test Loss: {test_loss}")

# exp.end()

file = open("History.txt", "a")
file.write(f"""
\n
MODEL | digit_recognition
model = Sequential()
model.add(Reshape(input_shape=(784,), target_shape=(28, 28, 1)))
model.add(Conv2D(kernel_size=(7, 7), filters=32, activation="relu"))
model.add(MaxPool2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, batch_size=64, epochs=25, validation_split=0.1)
Test Accuracy: {test_accuracy} Test Loss: {test_loss}""")
file.close()
