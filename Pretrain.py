# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:57:58 2018

@author: bear
"""
import argparse
import tensorflow as tf
import ILSVRC

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps',default=15000, type=int,
                    help='number of training steps')

parser.add_argument('--train_path', default='D:\ILSVRC2012_devkit_t12\data\TrainLabel_shuffled.csv', 
                    type=str, help='path of the train csv file')
parser.add_argument('--valid_path', default='D:\ILSVRC2012_devkit_t12\data\ValidSet.csv',
                    type=str, help='path of the valid csv file')
parser.add_argument('--test_path', default='D:\ILSVRC2012_devkit_t12\data\TestSet.csv', 
                    type=str, help='path of the test csv file')

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # ILSVRC images are 224x224 pixels, and have three color channel
  input_layer = tf.reshape(features, [-1, 224, 224, 3, 1])

  # Convolutional Layer #1
  # Computes 64 features using a 7x7 filter with Leaky-ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 224, 224, 3]
  # Output Tensor Shape: [batch_size, 112, 112, 3, 64]
  conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7, 1],
      strides=(2, 2, 1),
      padding="same",
      activation=my_leaky_relu)
  
  # reshape conv1 into : [batch_size, 112, 112, 192]
  conv1_reshaped = tf.reshape(
          conv1,
          (-1, 112, 112, 192))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 112, 112, 192]
  # Output Tensor Shape: [batch_size, 56, 56, 192]
  pool1 = tf.layers.max_pooling2d(inputs=conv1_reshaped , pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 192]
  # Output Tensor Shape: [batch_size, 56, 56, 256]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #3
  # Computes 128 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #4
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 128]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #5
  # Computes 256 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #6
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Pooling Layer #3
  # Third max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 14, 14, 512]
  pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #7
  # Computes 256 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 512]
  # Output Tensor Shape: [batch_size, 14, 14, 256]
  conv7 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #8
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 256]
  # Output Tensor Shape: [batch_size, 14, 14, 512]
  conv8 = tf.layers.conv2d(
      inputs=conv7,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #9
  # repeat convolutional Layer #7
  conv9 = tf.layers.conv2d(
      inputs=conv8,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #10
  # repeat convolutional Layer #9
  conv10 = tf.layers.conv2d(
      inputs=conv9,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
    # Convolutional Layer #11
  # repeat convolutional Layer #7
  conv11 = tf.layers.conv2d(
      inputs=conv10,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #12
  # repeat convolutional Layer #9
  conv12 = tf.layers.conv2d(
      inputs=conv11,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #13
  # repeat convolutional Layer #7
  conv13 = tf.layers.conv2d(
      inputs=conv12,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #14
  # repeat convolutional Layer #9
  # Output Tensor Shape: [batch_size, 14, 14, 512]
  conv14 = tf.layers.conv2d(
      inputs=conv13,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #15
  # Computes 512 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 512]
  # Output Tensor Shape: [batch_size, 14, 14, 512]
  conv15 = tf.layers.conv2d(
      inputs=conv14,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #16
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 512]
  # Output Tensor Shape: [batch_size, 14, 14, 1024]
  conv16 = tf.layers.conv2d(
      inputs=conv15,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Pooling Layer #4
  # Fourth max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 1024]
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  pool4 = tf.layers.max_pooling2d(inputs=conv16, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #17
  # Computes 512 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 1024]
  # Output Tensor Shape: [batch_size, 7, 7, 512]
  conv17 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #18
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 512]
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  conv18 = tf.layers.conv2d(
      inputs=conv17,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #19
  # Repeat Convolutional Layer # 17
  conv19 = tf.layers.conv2d(
      inputs=conv18,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #20
  # Repeat Convolutional Layer # 18
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  conv20 = tf.layers.conv2d(
      inputs=conv19,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Pooling Layer #5
  # Fourth max pooling layer with a 7x7 filter and stride of 1
  # Input Tensor Shape: [batch_size, 7, 7, 1024]
  # Output Tensor Shape: [batch_size, 1, 1, 1024]
  pool5 = tf.layers.average_pooling2d(inputs=conv20, pool_size=[7, 7], strides=1, padding='valid')


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 1, 1, 1024]
  # Output Tensor Shape: [batch_size, 1 * 1 * 1024]
  pool5_flat = tf.reshape(pool5, [-1, 1024])
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs= pool5_flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 364]
  logits = tf.layers.dense(inputs=dropout, units=364)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
#  eval_metric_ops = {
#      "accuracy": tf.metrics.accuracy(
#          labels=labels, predictions=predictions["classes"])}
  eval_metric_ops = {
      "accuracy": tf.metrics.mean(tf.nn.in_top_k(
          predictions=logits, targets=labels, k = 5))}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
  # Load training and eval data
  args = parser.parse_args(argv[1:])
  (train_filenames, train_labels), (valid_filenames, valid_labels), (test_filenames, test_labels
  ) = ILSVRC.load_data(args.train_path,args.valid_path,args.test_path)

  # Create the Estimator
  ILSVRC_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:/YOLO/YOLO_pretrain_model")

  # Train the model
  ILSVRC_classifier.train(
      input_fn=lambda:ILSVRC.train_input_fn(train_filenames, train_labels, args.batch_size),
      steps=args.train_steps)
  
#  conv_vars = [var for var in ILSVRC_classifier.get_variable_names()
#               if var.find('conv') != -1 and var.find('Adam') == -1]  
  # Evaluate the model and print results
  eval_result = ILSVRC_classifier.evaluate(
      input_fn=lambda:ILSVRC.eval_input_fn(test_filenames, test_labels,   args.batch_size))
  
  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)