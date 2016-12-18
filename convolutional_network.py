#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:20:20 2016

@author: Klemens
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy

from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import pickle
from skimage import transform
import cv2
import matplotlib.image as mpimg


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def gray_images(image_list, enable_plot = False):
    """
    applyies graysacle transformation to a list of images (numpy arrays)
    """
    grayed_image_list = []
    for image in image_list:
        gray = grayscale(image.astype(np.uint8))
        grayed_image_list.append(gray)
        
        if enable_plot:
            fig, axes = plt.subplots(1,2)
            axes[0].imshow(image)
            axes[1].imshow(gray, cmap='gray')
            plt.show()
            
    return np.array(grayed_image_list)
    
def shear_images(image_list, shear_factor = 0.2, enable_plot = False):
    """
    sheares a list of images (numpy arrays) and returns a new list of sheared images
    """
    sheared_image_list = []
    afine_tf = transform.AffineTransform(shear=shear_factor)
    for image in image_list:
        modified = transform.warp(image, afine_tf)
        sheared_image_list.append(modified)
        
        if enable_plot:
            fig, axes = plt.subplots(1,2)
            axes[0].imshow(image)
            axes[1].imshow(modified)
            plt.show()
        
    return np.array(sheared_image_list)
    
def load_data(training_file, testing_file, load_as_gray=False):
    """ 
    loads data from pickle
    divides training data and set used for training and set used for validation
    returns training, validation and test sets (images and corresponding classes)
    """
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train_all, y_train_all = train['features'], train['labels']
    n_train_all = X_train_all.shape[0]
    
    # rearrange data
    perm = np.arange(n_train_all)
    np.random.shuffle(perm)
    n_train = round(n_train_all*0.9)
    
    # Get the train images from the overall set.
    X_train = X_train_all[perm][:n_train]
    # Get the validation images from the overall set.
    y_train = y_train_all[perm][:n_train]
    
    # Get the train images from the overall set.
    X_val = X_train_all[perm][n_train:]
    # Get the validation images from the overall set.
    y_val = y_train_all[perm][n_train:]
    
    X_test, y_test = test['features'], test['labels']

    if load_as_gray:
        return gray_images(X_train), y_train, gray_images(X_val), y_val, gray_images(X_test), y_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test

def plot_all_classes(X_data, y_data):
    """
    plots all classes to get an overview
    """
    u,indices = np.unique(y_data, return_index=True)
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(7,7, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i, ax in enumerate(axes.flat):

        # Plot image.
        if i < len(indices):
            ax.imshow(X_data[indices][i])
    
            xlabel = "True: {0}".format(y_data[indices][i])
            
            # Show classes as label on x-axis
            ax.set_xlabel(xlabel)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    
    plt.show()
    
    
def plot_images(images, cls_true, cls_pred=None):
    """
    plot 9 images with true classes
    """
    assert len(images) == len(cls_true) <= 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i])#.reshape(image_shape))#, cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show classes as label on x-axis
        ax.set_xlabel(xlabel)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

    
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    # Create new biases, one for each filter.
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
    
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    shape = [num_inputs, num_outputs]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
def next_batch(X_data, y_data, mini_batch_size):

    # Get some random images
    n_data = X_data.shape[0]

    perm = np.arange(n_data)
    np.random.shuffle(perm)
    
    # Get the first images from the test-set.
    images_rand = X_data[perm][:mini_batch_size]

    # Get the true classes for those images.
    cls_true_rand = y_data[perm][:mini_batch_size]
    
    #shuffle data.x and data.y while retaining relation
    #return [data.x[:mini_batch_size], data.y[:mini_batch_size]]
    return [images_rand, cls_true_rand]

def dense_to_one_hot(labels_dense, num_classes=10):
  """
  Convert class labels from scalars to one-hot vectors
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

  return labels_one_hot
  
def optimize(X_train, y_train_hot_encoded, train_batch_size, num_iterations):
    # Ensure we update the global variable rather than a local copy.
#    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

#    for i in range(total_iterations,
#                   total_iterations + num_iterations):
    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = next_batch(X_train, y_train_hot_encoded, train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x_image: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
#    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
def plot_example_errors(X_test, y_test, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = X_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = y_test[incorrect]
    
    if len(cls_pred)>=9:
        # Plot the first 9 images.
        plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])
    else:
        for i, image in enumerate(images):
            plt.imshow(image)#.reshape(image_shape))#, cmap='binary')

            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show classes as label on x-axis
            plt.xlabel(xlabel)
            
            # Remove ticks
            plt.xticks([])
            plt.yticks([])
            plt.show()
    
def plot_confusion_matrix(y_test, cls_pred, n_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = y_test# data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    #plt.figure(figsize=(16, 8))
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the confusion matrix as an image.
    cbar = ax.matshow(cm)

    # Make various adjustments to the plot.
    fig.colorbar(cbar)
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks, range(n_classes))
    ax.set_yticks(tick_marks, range(n_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    


def print_test_accuracy(X_test, y_test, y_test_hot_encoded, y_pred_cls, n_classes, show_example_errors=False,
                        show_confusion_matrix=False):

    """
    determine accuracy of conv net when test set is used
    """
    # Split the test-set into smaller batches of this size.
    test_batch_size = 256
    
    # Number of images in the test-set.
    num_test = len(X_test)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = X_test[i:j, :]

        # Get the associated labels.
        labels = y_test_hot_encoded[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x_image: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = y_test

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} (correct: {1}/ tested: {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(X_test, y_test, cls_pred, correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(y_test, cls_pred, n_classes)

        
def resize_image(image_name, enable_plot = False):
    image = mpimg.imread(image_name)
    
    image_resized = scipy.misc.imresize(image, (32, 32))
    if enable_plot:
        plt.imshow(image_resized)
    return image_resized

if __name__ == "__main__":
    
    # load data
    training_file = 'lab 2 data/train.p'
    testing_file = 'lab 2 data/test.p'
    
    
    X_train_base, y_train_base, X_val, y_val, X_test, y_test = load_data(training_file, testing_file, load_as_gray = False)
    
    #X_train_sheared_pos = shear_images(X_train_base, 0.1)
    #X_train_sheared_neg = shear_images(X_train_base, -0.2)
    
    # numpy array will come as float. as imshow results in strange behaviour, with float, convert it back to unsigned int
    X_train = np.uint8(np.concatenate((X_train_base, X_train_base), axis=0))
    y_train = np.uint8(np.concatenate((y_train_base, y_train_base)))
    #X_train = X_train_base
    #y_train = y_train_base
    
    ### To start off let's do a basic data summary.
    # TODO: number of training examples
    n_train = X_train.shape[0]
    
    # TODO: number of testing examples
    n_test = X_test.shape[0]
    
    # TODO: what's the shape of an image?
    image_shape = X_train.shape[1:3]
    
    # TODO: how many classes are in the dataset
    n_classes = len(np.unique(y_train))
    
    # number of pixels in each dimension
    img_size = X_train.shape[1]
    
    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size
    
    # Number of colour channels for the images
    num_channels = X_train.shape[3]

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Image shape flattened =", img_size_flat)
    print("Number of channels = ", num_channels)
    print("Number of classes =", n_classes)
    
    y_train_hot_encoded = dense_to_one_hot(y_train, n_classes)
    y_val_hot_encoded = dense_to_one_hot(y_val, n_classes)
    y_test_hot_encoded = dense_to_one_hot(y_test, n_classes)

    # Get the first images from the test-set.
    images = X_train[0:9]

    # Get the true classes for those images.
    cls_true = y_train[0:9]

    # Plot all different classes to get an overview
    plot_all_classes(X_train, y_train)
    
    # Plot the images and labels using our helper-function above.
    plot_images(images=images, cls_true=cls_true)

    # Get some random images
    perm = np.arange(n_train)
    np.random.shuffle(perm)
    
    # Get the first images from the test-set.
    images_rand = X_train[perm][0:9]

    # Get the true classes for those images.
    cls_true_rand = y_train[perm][0:9]

    # Plot the images and labels using our helper-function above.
    plot_images(images=images_rand, cls_true=cls_true_rand)
    
    
    ### settings for network
    # Convolutional Layer 1.
    filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16         # There are 16 of these filters.
    
    # Convolutional Layer 2.
    filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 36         # There are 36 of these filters.
    
    # Fully-connected layer.
    fc_size = 128             # Number of neurons in fully-connected layer.
    

    ### Placeholder variables    
    x_image = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels], name='x_image')
    
    y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
    
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    
    ### Convolutional Layer 1
    # - takes x_image as input and creates num_filters1 different filters, 
    #   each having width and height equal to filter_size1
    # - down-sample the image so it is half the size by using 2x2 max-pooling
    layer_conv1, weights_conv1 = new_conv_layer(input = x_image,
                   num_input_channels = num_channels,
                   filter_size = filter_size1,
                   num_filters = num_filters1,
                   use_pooling = True)
    
    ### Convolutional Layer 2
    # - takes as input the output from the first convolutional layer
    # - number of input channels corresponds to number of filters in first convolutional layer.
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    
    ### Flatten layer
    layer_flat, num_features = flatten_layer(layer_conv2)
    
    # Fully-connected layer 1
    # - input is the flattened layer from the previous convolution
    # - number of neurons or nodes in fully-connected layer is fc_size
    layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
    
    # Fully-connected layer 2
    # - another fully-connected layer that outputs vectors of length 10 
    # for determining which of the 10 classes the input image belongs to. 
    # - ReLU is not used in this layer
    layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=n_classes,
                         use_relu=False)
    
    # extract single class allocation from neural network
    y_pred = tf.nn.softmax(layer_fc2)
    
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    # calculate costs
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    
    cost = tf.reduce_mean(cross_entropy)
    
    # initialize optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    
    # set up accuracy calculation
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ### Tensor Flow Run
    session = tf.Session()
    
    session.run(tf.initialize_all_variables())
    
    # split training data into batches that are then feeded into the network
    train_batch_size = 64
    
    # run optimization
    optimize(X_train, y_train_hot_encoded, train_batch_size, 1000)

    print('Validation Set')
    print_test_accuracy(X_val, y_val, y_val_hot_encoded, y_pred_cls, n_classes, show_example_errors=True, show_confusion_matrix=True)

    print('Test Set')
    print_test_accuracy(X_test, y_test, y_test_hot_encoded, y_pred_cls, n_classes, show_example_errors=True, show_confusion_matrix=True)

    #session.close()
    
    dict_traffic_signs = {
                          'traffic_signs_images_square/einbahnstrasse_17_1.png': 17,
                          'traffic_signs_images_square/einbahnstrasse_17.png': 17,
                          'traffic_signs_images_square/vorfahrtgewaehren_13_1.png': 13,
                          'traffic_signs_images_square/vorfahrtgewaehren_13.png': 13,
                          'traffic_signs_images_square/vorfahrtsschild_12_1.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_2.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_3.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_4.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12_5.png': 12,
                          'traffic_signs_images_square/vorfahrtsschild_12.png': 12,
                          }
    
    image_list_web = []
    label_list_web = []

    for image_name, label in dict_traffic_signs.items():        
        image = resize_image(image_name)
        image_list_web.append(image)
        label_list_web.append(label)
    
    X_web = np.array(image_list_web)
    y_web = np.array(label_list_web)
    y_web_hot_encoded = dense_to_one_hot(y_web, n_classes)

    
    print('Set from the web')
    print_test_accuracy(X_web, y_web, y_web_hot_encoded, y_pred_cls, n_classes, show_example_errors=False, show_confusion_matrix=False)
