import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

test = "./data/test/"
IMG_HEIGHT = IMG_WIDTH = 100

model = keras.models.load_model('./model/model')

data = ImageDataGenerator(rescale = 1./255)

test_generator = data.flow_from_directory(
        directory = test,
        target_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size = 64,
        )

model.evaluate(test_generator)
