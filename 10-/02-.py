from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from random import randint

def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
            Understanding the difficulty of training deep feedforward neural
            networks. International conference on artificial intelligence and
            statistics.
    Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
        An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


mnist = input_data.read_data_sets("./mnist/input_data", one_hot=True)

# parameters
learning_rate = 0.001
training_epoches = 15
batch_size = 100
display_step = 1

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W1 = tf.get_variable('W1', shape=[784, 256], initializer=xavier_init(784,256))
W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_init(256,256))
W3 = tf.get_variable('W3', shape=[256, 10], initializer=xavier_init(256,10))

B1 = tf.Variable(tf.zeros([256]))
B2 = tf.Variable(tf.zeros([256]))
B3 = tf.Variable(tf.zeros([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
hypothesis = tf.add(tf.matmul(L2, W3), B3)

#Construct model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) # Cross entropy
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)	# Gradient Descent

init = tf.global_variables_initializer()

#launch the graph
with tf.Session() as sess:
    sess.run(init)

    #training cycle
    for epoch in range(training_epoches):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y:batch_ys})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y:batch_ys}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # r = randint(0, mnist.test.num_examples-1)
    # print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # print('Prediction: ', sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]}))

    # plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()

    print("Optimization Finished!")

	#Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
	#Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))