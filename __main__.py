from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

import csv
import random

# create model with 4 inputs and 3 outputs
model = Sequential()

model.add(Dense(4, input_shape=(4,)))
model.add(Activation("sigmoid"))

model.add(Dense(8))
model.add(Activation("sigmoid"))

model.add(Dense(3))
model.add(Activation("sigmoid"))

# use loss function for multiple classes
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# read dataset
data = []
data_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

with open("iris.data", "r") as iris_data:
    iris_reader = csv.reader(iris_data, delimiter=",")
    for sample in iris_reader:
        if len(sample) == 5:
            data.append(sample)

random.shuffle(data)
half = int(len(data) * 0.5)

train_data = []
train_labels = []

for sample in data[:half]:
    sample_data = sample[:4]
    sample_label = [0, 0, 0]
    sample_label[data_classes.index(sample[4])] = 1
    train_data.append(sample_data)
    train_labels.append(sample_label)

test_data = []
test_labels = []

for sample in data[half:]:
    sample_data = sample[:4]
    sample_label = [0, 0, 0]
    sample_label[data_classes.index(sample[4])] = 1
    test_data.append(sample_data)
    test_labels.append(sample_label)

# train model
BATCH_SIZE = 12
for i in range(int(len(train_data) / BATCH_SIZE)):
    batch_size = BATCH_SIZE
    lower = i
    upper = i + batch_size
    if len(train_data) - 1 < upper:
        upper = len(train_data) - 1
        batch_size = len(train_data) - 1 - i
    batch_data = np.array(train_data[lower:upper])
    batch_labels = np.array(train_labels[lower:upper])
    model.fit(batch_data, batch_labels, epochs=1000, batch_size=batch_size)


print("Evaluating model...")

# evaluate model
BATCH_SIZE = 12
for i in range(int(len(test_data) / BATCH_SIZE)):
    batch_size = BATCH_SIZE
    lower = i
    upper = i + batch_size
    if len(test_data) - 1 < upper:
        upper = len(test_data) - 1
        batch_size = len(test_data) - 1 - i
    batch_data = np.array(test_data[lower:upper])
    batch_labels = np.array(test_labels[lower:upper])
    loss, accuracy = model.evaluate(batch_data, batch_labels, batch_size=batch_size)
    print("loss: {0} - accuracy: {1}".format(loss, accuracy))
