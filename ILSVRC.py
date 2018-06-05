import pandas as pd
import tensorflow as tf
import cv2
def load_data(train_path, valid_path, test_path):
    """Returns the ILSVRC dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv(train_path, header=None)
    valid = pd.read_csv(valid_path, header=None)
    test  = pd.read_csv( test_path, header=None)
    
    train_filenames = train[0]
    train_labels    = train[1]
    
    valid_filenames = valid[0]
    valid_labels    = valid[1]
    
    test_filenames = test[0]
    test_labels    = test[1]
    
    train_filenames = tf.constant(train_filenames)
    train_labels    = tf.constant(train_labels   )
    valid_filenames = tf.constant(valid_filenames)
    valid_labels    = tf.constant(valid_labels   )
    test_filenames  = tf.constant(test_filenames )
    test_labels     = tf.constant(test_labels    )
    
    return (train_filenames, train_labels), (valid_filenames, valid_labels), (test_filenames, test_labels)


def train_input_fn(filenames, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
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
def _read_py_function(filename, label):
#  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_COLOR)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  return image_decoded, label

# Use standard TensorFlow operations to normalize the image to a fixed shape.
def _normalize_function(image_decoded, label):
  image_normalized = tf.cast(image_decoded, tf.float32) * (1. / 255) - 0.5
  return image_normalized, label