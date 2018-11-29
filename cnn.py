import numpy as np
import tensorflow as tf
import image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Read train data
train_n, train_p = image.readImages("E:/Machine Learning/new/train")
train = train_n + train_p
print("Read train data done")

# Create train labels
train_labels = []
for i in range(len(train_n)):
    train_labels.append([0])
for i in range(len(train_p)):
    train_labels.append([1])

# # Read test data
# test_n, test_p = image.readImages("E:/Machine Learning/chest_xray/test")
# test = test_n + test_p
# print("Read test data done")
#
# # Create test labels
# test_labels = []
# test_labels += len(test_n) * [0]
# test_labels += len(test_p) * [1]


def weight(shape):
    weight = tf.Variable(tf.random_normal(shape, stddev=0.1))
    return weight


def bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def conv(x, W, strides):
    return tf.nn.conv2d(x, W, strides, padding='SAME')


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 10000])
y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 100, 100, 1])

# layer 1
weight_1 = weight([5, 1, 1, 32])
bias_1 = bias([32])
conv_1 = conv(x_image, weight_1, [1, 1, 1, 1])
relu_1 = tf.nn.relu(conv_1 + bias_1)
pool_1 = pool(relu_1)

# layer 2
weight_2 = weight([5, 5, 32, 64])
bias_2 = bias([64])
conv_2 = conv(pool_1, weight_2, [1, 1, 1, 1])
relu_2 = tf.nn.relu(conv_2 + bias_2)
pool_2 = pool(relu_2)

# fully connected layer
pool_2_fc = tf.reshape(pool_2, [-1, 7 * 7 * 64])
weight_fc = weight([7 * 7 * 64, 1024])
bias_fc = bias([1024])
fc = tf.nn.relu(tf.matmul(pool_2_fc, weight_fc) + bias_fc)

weight_fc2 = weight([1024, 2])
bias_fc2 = bias([2])
prediction = tf.nn.softmax(tf.matmul(fc, weight_fc2) + bias_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_step, feed_dict={
                 x: train, y: train_labels, keep_prob: 0.5})
