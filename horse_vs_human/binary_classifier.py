import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# input the data to program
train_horse_dir = 'data/horse_or_human/horses'
train_human_dir = 'data/horse_or_human/humans'

validation_horse_dir = 'data/validation_horse_or_human/horses'
validation_human_dir = 'data/validation_horse_or_human/humans'

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

validation_horse_hames = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)


def display_data():
    # output images in a 4x4 configuration
    nrows, ncols = 4, 4
    # Index for iterating over images
    pic_index = 0

    # set up matplotlib fig, & size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_horse_pix+next_human_pix):
        # set up subplots; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

# display_data()



def model():
    # define model layers
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(300, 300, 3)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'), 
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # define model loss & optimizers
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    return model


def data_generator():
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        directory='data/horse_or_human/',       #the source directory for training images
        target_size=(300,300),                  #All images will be resized to 300x300
        batch_size=64,
        class_mode='binary'                     #Since we use binary_crossentropy loss, we need binary labels
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory='data/validation_horse_or_human/',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary'
    )

    return train_generator, validation_generator


def visualize_intermediate_representation(model):

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    visualization_model = keras.models.Model(inputs = model.input, outputs = successive_outputs)
    # Let's prepare a random input image from the training set.
    horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
    human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
    img_path = random.choice(horse_img_files + human_img_files)

    img = image.load_img(img_path, target_size=(300, 300))  # this is a PIL image
    x = image.img_to_array(img)  # Numpy array with shape (300, 300, 3)
    print(x.shape)
    x = np.expand_dims(x, axis=0) # Numpy array with shape (1, 300, 300, 3)
    print(x.shape)

    # Rescale by 1/255
    x /= 255

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x = x / x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
                # Display the grid
                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()



model = model()

# print(model.summary())
train_generator, validation_generator = data_generator()
# print('train')
# for i in train_generator:
#     print(i)
#     break
# exit()

''' training the model'''
model.fit(
    train_generator,
    steps_per_epoch=8,      #8*128=1024 completing a epoch
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
    )

# visualize_intermediate_representation(model)
# exit()

dir_path = 'data/test'
list_img = os.listdir(dir_path)
for img_name in list_img:
    img_path = os.path.join(dir_path, img_name)
    img = image.load_img(img_path, target_size=(300, 300))
    x = image.img_to_array(img)
    # print('shapes')
    # print(x.shape)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)
    images = np.vstack([x])
    # print('images', images.shape)
    classes = model.predict(images, batch_size=10)
    if classes[0]>0.5:
        print(img_name + " is a human")
    else:
        print(img_name + " is a horse")