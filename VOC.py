import pandas as pd
import tensorflow as tf
import numpy as np

def train_load_data(train_csv_path, train_npz_path):
    """Returns the ILSVRC dataset as (train_x, train_y), (test_x, test_y)."""

    train_csv = pd.read_csv(train_csv_path, header=None)
   
    train_filenames = train_csv[1]
    npzfile = np.load(train_npz_path)
    
    object_classes = npzfile['object_classes']
    locations = npzfile['locations']
    sizes = npzfile['sizes']
    coordinates = npzfile['coordinates']
    
    train_filenames = tf.constant(train_filenames)
    object_classes  = tf.constant(object_classes , dtype=tf.float32)
    locations       = tf.constant(locations      , dtype=tf.bool)
    sizes           = tf.constant(sizes          , dtype=tf.float32)
    coordinates     = tf.constant(coordinates    , dtype=tf.float32)
    
    return train_filenames, object_classes, locations, sizes, coordinates


def train_input_fn(train_filenames, object_classes, locations, sizes, coordinates, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((train_filenames, object_classes, locations, sizes, coordinates))
    # Parse each line.
#    dataset = dataset.map(
#        lambda filename, label: tuple(tf.py_func(
#                _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_read_py_function)
    dataset = dataset.map(_normalize_function)
    # Shuffle, repeat, and batch the examples.

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000).batch(batch_size)
    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()



def eval_input_fn(filenames, labels, batch_size):
    """An input function for evaluation"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Parse each line.
#    dataset = dataset.map(
#        lambda filename, label: tuple(tf.py_func(
#                _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_read_py_function)
    dataset = dataset.map(_normalize_function)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()

def test_input_fn(filenames, labels, batch_size):
    """An input function for testing"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Parse each line.
#    dataset = dataset.map(
#        lambda filename, label: tuple(tf.py_func(
#                _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_read_py_function)
    dataset = dataset.map(_normalize_function)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset.make_one_shot_iterator().get_next()

# The remainder of this file contains a simple example of a Image parser,
#     implemented using the `Dataset` class.

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(train_filenames, object_classes, locations, sizes, coordinates):
#  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_COLOR)
  image_string = tf.read_file(train_filenames)
  image_decoded = tf.image.decode_jpeg(image_string)
  return image_decoded, object_classes, locations, sizes, coordinates

# Use standard TensorFlow operations to normalize the image to a fixed shape.
def _normalize_function(image_decoded, object_classes, locations, sizes, coordinates):
  image_normalized = tf.cast(image_decoded, tf.float32) * (1. / 255) - 0.5
  return image_normalized, (object_classes, locations, sizes, coordinates)