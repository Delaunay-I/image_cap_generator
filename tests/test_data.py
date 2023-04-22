import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_num_images():
    jpgs = os.listdir("./flickr8k/Images")
    assert len(jpgs) == 8091