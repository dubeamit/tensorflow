import tensorflow as tf
import urllib.request
import os
import zipfile
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy as np


def create_dirs():
    os.makedirs('data/cats_vs_dogs/training/cats', exist_ok=True)
    os.makedirs('data/cats_vs_dogs/training/dogs', exist_ok=True)
    os.makedirs('data/cats_vs_dogs/testing/cats', exist_ok=True)
    os.makedirs('data/cats_vs_dogs/testing/dogs', exist_ok=True)

create_dirs()

def download_data():
    data_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
    data_file_name = 'catsdogs.zip'
    download_dir = 'data'
    urllib.request.urlretrieve(data_url, data_file_name)
    zip_ref = zipfile.ZipFile(data_file_name, 'r')
    zip_ref.extractall(download_dir)
    zip_ref.close()

# download_data()

# move the files from the extracted dir to train/test dir
def move_files(files, source, dest):
    for filename in files:
        this_file = os.path.join(source, filename)
        destination = os.path.join(dest, filename)
        copyfile(this_file, destination)


# split the train/test data
def split_data(source, training, testing, split_size):
    files = []
    for filename in os.listdir(source):
        file = os.path.join(source, filename)
        # ignore if file is empty
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(f'{filename} is zero length, so ignoring.')

    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:]
    
    move_files(training_set, source, training)
    move_files(testing_set, source, testing)

    
cat_source_dir = 'data/PetImages/Cat'
training_cats_dir = 'data/cats_vs_dogs/training/cats'
testing_cats_dir = 'data/cats_vs_dogs/testing/cats'

dog_source_dir = 'data/PetImages/Dog'
training_dogs_dir = 'data/cats_vs_dogs/training/dogs'
testing_dogs_dir = 'data/cats_vs_dogs/testing/dogs'

split_size = 0.9
if False:
    split_data(cat_source_dir, training_cats_dir, testing_cats_dir, split_size)
    split_data(dog_source_dir, training_dogs_dir, testing_dogs_dir, split_size)

    print("Number of training cat images", len(os.listdir(training_cats_dir)))
    print("Number of training dog images", len(os.listdir(training_dogs_dir)))
    print("Number of testing cat images", len(os.listdir(testing_cats_dir)))
    print("Number of testing dog images", len(os.listdir(testing_dogs_dir)))


# use data augmentations built in ImageDataGenerator
def data_augmentation():
    training_dir = 'data/cats_vs_dogs/training'
    training_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_generator = training_datagen.flow_from_directory(
        training_dir,
        batch_size=100,
        class_mode='binary',
        target_size=(150,150)
    )
    # visualize the data
    # for i in train_generator:
    #     img = next(train_generator)[0]
    #     plt.imshow(img[0])
    #     plt.show()
    #     # break

    validation_dir = 'data/cats_vs_dogs/testing/'
    validation_datagen = ImageDataGenerator(rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=100,
        class_mode='binary',
        target_size=(150,150)
    )

    return train_generator, validation_generator

train_generator, validation_generator = data_augmentation()


# get the pretrained weights & freeze the cnn layers
def download_pretrained_model():
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)

    #Instantiate the model
    pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )

    # load pre-trained weights
    pre_trained_model.load_weights(weights_file)

    # freeze the layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # pre_trained_model.summary()

    # get a reference to the last layer, 'mixed7' because we'll add some layers after this last layer
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output

    return pre_trained_model.input, last_output


# create our on FC layers after the last cnn layers
def model(pre_trained_input, last_output):
    # Flatten the output layer to 1D
    x = layers.Flatten()(last_output)
    # Add a FC layer with 1024 hidden units & ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_input, x)

    # compile the model
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def fit(model):
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=2,
        verbose=1
    )
    return history

pre_trained_input, last_output = download_pretrained_model()
model = model(pre_trained_input, last_output)
history = fit(model)

''' visualize the training and validation accuracy '''

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()


# predicting images
path = 'data/cats_vs_dogs/test_cat_dog/'
for img_name in path:
    img = load_img(os.path.join(path, img_name), target_size=(150, 150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    if classes[0] > 0.5:
        print(f'{img_name} is a dog')
    else:
        print(f'{img_name} is a cat')