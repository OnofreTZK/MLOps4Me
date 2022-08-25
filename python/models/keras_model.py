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
