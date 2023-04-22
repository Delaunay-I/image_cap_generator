from text_preprocessor import text_clean
from inception_encoder import InceptionEncoder

import numpy as np
import pytest
import os



def test_text_clean():
    raw_text = "I was s over there, a once came 3 people that.!?- a k"
    out_text = text_clean(raw_text)
    assert out_text == " was over there once came people that"

# Get the directory where this file is located
test_dir = os.path.dirname(os.path.abspath(__file__))

# Create an instance of the InceptionEncoder class for use in the tests
encoder = InceptionEncoder()

# Test the shape of the output encoded image
def test_encode_shape():
    image_path = os.path.join(test_dir, 'test_image.jpg')
    vec = encoder.encode(image_path)
    assert vec.shape == (2048,)

# Test that the output encoded image is a numpy array
def test_encode_output_type():
    image_path = os.path.join(test_dir, 'test_image.jpg')
    vec = encoder.encode(image_path)
    assert type(vec) == np.ndarray

# Test that calling preprocess and encode with the same image path produces the same output
def test_encode_consistency():
    image_path = os.path.join(test_dir, 'test_image.jpg')
    vec1 = encoder.encode(image_path)
    vec2 = encoder.image_model.predict(encoder.preprocess(image_path), verbose=0).flatten()
    assert np.array_equal(vec1, vec2)

# Test that calling encode with an invalid image path raises an error
def test_encode_error():
    with pytest.raises(Exception):
        encoder.encode('invalid_image.jpg')