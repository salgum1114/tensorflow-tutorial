from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import randint

mnist = input_data.read_data_sets("./mnist/input_data", one_hot=True)

# parameters
training_epoches = 25
display_step = 1
batch_size = 100
learning_rate = 0.01

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b)	#softmax

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices = 1)) # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)	# Gradient Descent

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
            sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    r = randint(0, mnist.test.num_examples-1)
    print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('Prediction: ', sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

	# print("Optimization Finished!")

	# #Test model
	# correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
	# #Calculate accuracy
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))