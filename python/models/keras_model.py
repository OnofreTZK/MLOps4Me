import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

import mlflow
import mlflow.tensorflow

(x_training, y_training), (x_test, y_test) = mnist.load_data()
# []_training[21] --> picking the pixels of the line 21
# 0
plt.imshow(x_training[21], cmap='gray')
plt.title(y_training[21])
plt.savefig("{}.png".format(y_training[21]))

# 3
plt.imshow(x_training[27], cmap='gray')
plt.title(y_training[27])
plt.savefig("{}.png".format(y_training[27]))

# 7
plt.imshow(x_training[29], cmap='gray')
plt.title(y_training[29])
plt.savefig("{}.png".format(y_training[29]))

x_training = x_training.reshape((len(x_training), np.prod(x_training.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_test[0])

x_training = x_training.astype('float32')
x_test = x_test.astype('float32')

x_training /= 255
x_test /= 255

y_training = np_utils.to_categorical(y_training, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(y_test[0])

def train_deep_learn(n_hidden_layers, n_units, activation, drop_out, epochs):
    mlflow.set_experiment("DLExperiment")

    with mlflow.start_run():

        mlflow.tensorflow.autolog()

        model = Sequential()

        # Create the hidden layer and the input layer
        model.add(Dense(units=n_units, activation=activation, input_dim=784))
        # input_dim -> number of neurons of the input label
        # 1 neuron per pixel
        # the array is 28x28 which has 784 pixels
        model.add(Dropout(drop_out))

        # aditional hidden layers, with drop out
        for n in range(n_hidden_layers):
            model.add(Dense(units=n_units, activation=activation))
            model.add(Dropout(drop_out))

        # Output layer
        model.add(Dense(units=10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        historic = model.fit(x_training, y_training, epochs=epochs,
                             validation_data=(x_test, y_test))

        # graphic for errors and accuracy
        historic.history.keys()
        loss = plt.plot(historic.history['val_loss'])
        plt.savefig("../images/keras_loss.png")
        accuracy = plt.plot(historic.history['val_accuracy'])
        plt.savefig('../images/keras_accuracy.png')

        mlflow.log_artifact('../images/keras_loss.png')
        mlflow.log_artifact('../images/keras_accuracy.png')


        # execution info
        print("Model: ", mlflow.active_run().info.run_uuid)

    mlflow.end_run()

train_deep_learn(2, 16, 'relu', 0.2, 2,)
