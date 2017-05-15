# Import the converted model's class , solo funciona con python 2.7
from kaffe.tensorflow import Network
from tools import utils

import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

from tools import cifar10
# ------------------------ DATA DIMENSION ------------------
from tools.cifar10 import img_size, num_channels, num_classes

class AlexNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(10, relu=False, name='fc8')
             .softmax(name='prob'))
        self.dict= {
            "conv1" : True,
            "conv2" : True,
            "conv3" : True,
            "conv4": True,
            "conv5": True,
            "fc6": True,
            "fc7": False,
            "fc8": False
        }

if __name__=="__main__":
    # Set the path for storing the data-set on your computer.
    cifar10.data_path = "CIFAR_10_data/"

    cifar10.maybe_download_and_extract()

    class_names = cifar10.load_class_names()
    print(class_names)

    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    # The images are 32 x 32 pixels, but we will crop the images to 24 x 24 pixels.
    img_size_cropped = 24

    batch_size = 64

    # Placeholder variables
    images = tf.placeholder(tf.float32, [batch_size, img_size, img_size, num_channels])
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    y_true_cls = tf.argmax(labels, dimension=1)

    # Get the data specifications for the GoogleNet model

