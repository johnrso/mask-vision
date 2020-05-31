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

preds = model.predict(test_generator)

predicted_class_indices=np.argmax(preds,axis=1)
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print(predictions)
