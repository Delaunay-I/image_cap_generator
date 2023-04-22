from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

import numpy as np

class InceptionEncoder:
    def __init__(self):
        self.base_model = InceptionV3(weights='imagenet')
        self.image_model = Model(inputs=self.base_model.input, outputs=self.base_model.layers[-2].output)

    def preprocess(self, image_path):
        # inception v3 excepts img in 299 * 299 * 3
        image = load_img(image_path, target_size=(299, 299))
        # convert the image pixels to a numpy array
        x = img_to_array(image)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def encode(self, image_path):
        image = self.preprocess(image_path)
        vec = self.image_model.predict(image, verbose=0)
        vec_flattened = vec.flatten()
        return vec_flattened
