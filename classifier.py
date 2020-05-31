import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

train = "./data/train/"
test = "./data/test/"

IMG_HEIGHT = IMG_WIDTH = 100

data = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
        validation_split = .2
    )

train_generator = data.flow_from_directory(
        directory = train,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = 64,
        subset = "training"
        )

validation_generator = data.flow_from_directory(
        directory = train,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = 64,
        subset = "validation"
        )

model = tf.keras.Sequential([
    Conv2D(16, 3, padding = 'same', activation = 'relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dense(3)
])

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'])

model.summary()

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 2000 // 64,
        epochs = 20,
        validation_data = validation_generator,
        validation_steps = 800 // 64)

model.save_weights('weights.h5')  # always save your weights after training or during training
