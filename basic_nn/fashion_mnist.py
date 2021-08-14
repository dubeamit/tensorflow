import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time


# stop training when a desired accuracy is achieved
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.01:
            print(f'Reached 99% accuracy so cancelling training!')
            self.model.stop_training = True



callbacks = myCallback()
# Download & load fashion mnist dataset
mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_img, test_labels) = mnist.load_data()

# see the data
np.set_printoptions(linewidth=200)
plt.imshow(train_imgs[40000], cmap='gray')
print(train_labels[40000])
print(train_imgs[40000])
# plt.show()

train_imgs = train_imgs.reshape(60000, 28, 28, 1)
test_img = test_img.reshape(10000, 28, 28, 1)

# preprocess the data
train_imgs = train_imgs / 255.0
test_img = test_img / 255.0

# define the model
model = keras.models.Sequential([
                                keras.layers.Conv2D(1024, (5,5), activation='relu', input_shape=(28, 28, 1)),
                                keras.layers.MaxPooling2D((2, 2), (2,2)),
                                # keras.layers.Conv2D(64, (3,3), activation='relu'),
                                # keras.layers.MaxPooling2D((2, 2), (2,2)),
                                #  keras.layers.Conv2D(64, (3,3), activation='relu'),
                                # keras.layers.MaxPooling2D((2, 2), (2,2)),
                                keras.layers.Flatten(),
                                keras.layers.Dense(1024, activation=tf.nn.relu),
                                # keras.layers.Dense(128, activation=tf.nn.relu),
                                keras.layers.Dense(10, activation=tf.nn.softmax)])

# define loss & optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# train the model
t0 = time.time()
model.fit(train_imgs, train_labels, epochs=20, callbacks=[callbacks])
print(f'Time taken to train model {time.time()-t0} seconds')

# evaluate model on test data
loss, accuracy =model.evaluate(test_img, test_labels)

print(f'loss on test data {loss}')
print(f'accuracy on test data {accuracy}')

# predict on test
classifications = model.predict(test_img)
print(f'prediction {classifications[0]}')
print(f'label {test_labels[0]}')