# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:57:58 2018

@author: bear
"""
import argparse
import tensorflow as tf
import VOC

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=30, type=int, help='batch size')
parser.add_argument('--train_steps',default=1 , type=int,
                    help='number of training steps')

parser.add_argument('--train_csv_path', default= 'D:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\TrainSet.csv', 
                    type=str, help='path of the train csv file')
parser.add_argument('--train_npz_path', default= 'D:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\TrainSet.npz',
                    type=str, help='path of the train csv file')

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def IOU(sizes, sizes_est, coordinates, coordinates_est):
  xymax = coordinates + sizes/2;
  xymin = coordinates - sizes/2;

  xymax_est = coordinates_est + sizes_est/2;
  xymin_est = coordinates_est - sizes_est/2;

  xymax_i =  tf.minimum(xymax, xymax_est)   
  xymin_i =  tf.maximum(xymin, xymin_est)

  wh = xymax_i - xymin_i
  wh_clipped = tf.maximum(wh, 0)
  intersection = tf.multiply(wh_clipped[:,:,0], wh_clipped[:,:,1])
  union = tf.multiply(sizes[:,:,0], sizes[:,:,1]) + tf.multiply(sizes_est[:,:,0], sizes_est[:,:,1]) - intersection
  ov    = tf.divide(intersection, union)
  return ov


def cnn_model_fn(features, labels, mode):
  object_classes = labels[0]
  locations = labels[1]
  locations_float = tf.cast(locations, tf.float32)
  sizes = labels[2]
  coordinates = labels[3]
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # VOC images are 448x448 pixels, and have three color channel
  input_layer = tf.reshape(features, [-1, 448, 448, 3, 1])

  # Convolutional Layer #1
  # Computes 64 features using a 7x7 filter with Leaky-ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 448, 448, 3, 1]
  # Output Tensor Shape: [batch_size, 224, 224, 3, 64]
  conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7, 1],
      strides=(2, 2, 1),
      padding="same",
      activation=my_leaky_relu)
  
  # reshape conv1 into : [batch_size, 224, 224, 192]
  conv1_reshaped = tf.reshape(
          conv1,
          (-1, 224, 224, 192))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 224, 224, 192]
  # Output Tensor Shape: [batch_size, 112, 112, 192]
  pool1 = tf.layers.max_pooling2d(inputs=conv1_reshaped , pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 112, 112, 192]
  # Output Tensor Shape: [batch_size, 112, 112, 256]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 112, 112, 256]
  # Output Tensor Shape: [batch_size, 56, 56, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #3
  # Computes 128 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 256]
  # Output Tensor Shape: [batch_size, 56, 56, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #4
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 128]
  # Output Tensor Shape: [batch_size, 56, 56, 256]
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #5
  # Computes 256 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 256]
  # Output Tensor Shape: [batch_size, 56, 56, 256]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #6
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 56, 56, 256]
  # Output Tensor Shape: [batch_size, 56, 56, 512]
  conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Pooling Layer #3
  # Third max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #7
  # Computes 256 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  conv7 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #8
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
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
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  conv14 = tf.layers.conv2d(
      inputs=conv13,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #15
  # Computes 512 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 512]
  conv15 = tf.layers.conv2d(
      inputs=conv14,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #16
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 512]
  # Output Tensor Shape: [batch_size, 28, 28, 1024]
  conv16 = tf.layers.conv2d(
      inputs=conv15,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Pooling Layer #4
  # Fourth max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 1024]
  # Output Tensor Shape: [batch_size, 14, 14, 1024]
  pool4 = tf.layers.max_pooling2d(inputs=conv16, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #17
  # Computes 512 features using a 1x1 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 1024]
  # Output Tensor Shape: [batch_size, 14, 14, 512]
  conv17 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #18
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 512]
  # Output Tensor Shape: [batch_size, 14, 14, 1024]
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
  # Output Tensor Shape: [batch_size, 14, 14, 1024]
  conv20 = tf.layers.conv2d(
      inputs=conv19,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #21
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 1024]
  # Output Tensor Shape: [batch_size, 14, 14, 1024]
  conv21 = tf.layers.conv2d(
      inputs=conv20,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #22
  # Computes 1024 features using a 3x3 filter and stride of 2.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 1024]
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  conv22 = tf.layers.conv2d(
      inputs=conv21,
      filters=1024,
      kernel_size=[3, 3],
      strides=(2, 2),
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #23
  # Computes 1024 features using a 3x3 filter
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 1024]
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  conv23 = tf.layers.conv2d(
      inputs=conv22,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Convolutional Layer #24
  # Computes 1024 features using a 3x3 filter
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 7, 7, 1024]
  # Output Tensor Shape: [batch_size, 7, 7, 1024]
  conv24 = tf.layers.conv2d(
      inputs=conv23,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=my_leaky_relu)
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 1024]
  # Output Tensor Shape: [batch_size, 7 * 7 * 1024]
  conv24_flat = tf.reshape(conv24, [-1, 50176])
  
  # fully connected layer #1
  # Input Tensor Shape: [batch_size, 50176]
  # Output Tensor Shape: [batch_size, 4096]
  conn1 = tf.layers.dense(inputs=conv24_flat, units=4096, activation = my_leaky_relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs= conn1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
  
  # fully connected layer #2
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 1470]
  conn2 = tf.layers.dense(inputs=dropout, units=1470)
  
  # reshape tensor for our convenience
  # Input Tensor Shape: [batch_size, 1470]
  # Output Tensor Shape: [batch_size, 49, 30]
  conn2_reshaped = tf.reshape(conn2, [-1, 49, 30])
  
  probabilities = conn2_reshaped[:,:, 0:20]
  
  sizes_est1         = conn2_reshaped[:,:,20:22]
  sizes_est1_clipped = tf.maximum(sizes_est1, 0)
  sizes_est2         = conn2_reshaped[:,:,22:24]
  sizes_est2_clipped = tf.maximum(sizes_est1, 0)
  
  coordinates_est1   = conn2_reshaped[:,:,24:26]
  coordinates_est2   = conn2_reshaped[:,:,26:28]
  
  confidence_est1    = conn2_reshaped[:,:,28:29]
  confidence_est2    = conn2_reshaped[:,:,29:30]
  print(confidence_est2.shape)
  
  ov1 = IOU(sizes, sizes_est1_clipped, coordinates, coordinates_est1)
  ov1_expanded = tf.expand_dims(ov1, axis = 2)
  ov2 = IOU(sizes, sizes_est2_clipped, coordinates, coordinates_est2)
  ov2_expanded = tf.expand_dims(ov2, axis = 2)

  one_i1_obj   = tf.logical_and((ov1_expanded  > ov2_expanded), locations)
  one_i2_obj   = tf.logical_and((ov1_expanded <= ov2_expanded), locations)
  one_i1_noobj = tf.logical_not(one_i1_obj)
  one_i2_noobj = tf.logical_not(one_i2_obj)
  
  one_i1_obj_float = tf.cast(one_i1_obj, tf.float32)
  one_i2_obj_float = tf.cast(one_i2_obj, tf.float32)
  one_i1_noobj_float = tf.cast(one_i1_noobj, tf.float32)
  one_i2_noobj_float = tf.cast(one_i2_noobj, tf.float32)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "sizes_est1"      :   sizes_est1      ,
      "sizes_est2"      :   sizes_est2      ,
      "coordinates_est1":   coordinates_est1,
      "coordinates_est2":   coordinates_est2,
      "confidence_est1" :   confidence_est1 ,
      "confidence_est2" :   confidence_est2 ,
      "probabilities"   :   probabilities
  }
  if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for TRAIN modes)
  loss = 5*tf.reduce_sum(tf.multiply(tf.square(coordinates - coordinates_est1), one_i1_obj_float) + \
                         tf.multiply(tf.square(coordinates - coordinates_est2), one_i2_obj_float) + \
                         tf.multiply(tf.square(tf.sqrt(sizes) - tf.sqrt(sizes_est1_clipped)), one_i1_obj_float) + \
                         tf.multiply(tf.square(tf.sqrt(sizes) - tf.sqrt(sizes_est2_clipped)), one_i2_obj_float), axis=(1,2)) + \
           tf.reduce_sum(tf.multiply(tf.square(ov1_expanded - confidence_est1), one_i1_obj_float) + \
                         tf.multiply(tf.square(ov2_expanded - confidence_est2), one_i2_obj_float) + \
                     0.5*tf.multiply(tf.square(confidence_est1), one_i1_noobj_float) + \
                     0.5*tf.multiply(tf.square(confidence_est2), one_i2_noobj_float), axis=(1,2)) + \
           tf.reduce_sum(tf.multiply(tf.square(probabilities - object_classes), locations_float), axis=(1,2))
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def main(argv):
  # Load training and eval data
  args = parser.parse_args(argv[1:])
  train_filenames, object_classes, locations, sizes, coordinates = VOC.train_load_data(args.train_csv_path, args.train_npz_path)

  # Create the Estimator
  VOC_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:/YOLO/YOLO_model")

  # Train the model
  VOC_classifier.train(
      input_fn=lambda:VOC.train_input_fn(train_filenames, object_classes, locations, sizes, coordinates, args.batch_size),
      steps=args.train_steps)
  
#  conv_vars = [var for var in VOC_classifier.get_variable_names()
#               if var.find('conv') != -1 and var.find('Adam') == -1]  
  # Evaluate the model and print results
#  eval_result = VOC_classifier.evaluate(
#      input_fn=lambda:VOC.eval_input_fn(valid_filenames, valid_labels,   args.batch_size))
#  
#  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)