import pickle
import numpy as np
import matplotlib.pyplot as plt

train_balanced_data_file_preprocessed = "traffic-signs-data/train_balanced_preprocessed.p"
test_data_file_preprocessed = "traffic-signs-data/test_preprocessed.p"
valid_data_file_preprocessed = "traffic-signs-data/valid_preprocessed.p"

def load_data(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    X, y = data['features'], data['labels']
    return X, y

def normalize(data): 
    data = (data - data.min()) / (data.max() - data.min())
    return data

X_train, y_train = load_data(train_balanced_data_file_preprocessed)
X_valid, y_valid = load_data(valid_data_file_preprocessed)
X_test, y_test = load_data(test_data_file_preprocessed)
    
X_train = normalize(X_train)
X_valid = normalize(X_valid)
X_test = normalize(X_test)

# Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    # Activation.
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def LeNet(x, is_training):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    output_size = 43
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    w1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    b1 = tf.Variable(tf.zeros(32))
    conv1 = conv2d(x, w1, b1)

    # Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.cond(is_training, lambda: tf.nn.dropout(conv1, keep_prob = 0.9), lambda: conv1)

    # Layer 2: Convolutional. Output = 10x10x64.
    w2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    b2 = tf.Variable(tf.zeros(64))
    conv2 = conv2d(conv1, w2, b2)

    # Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = maxpool2d(conv2, k=2)
    conv2 = tf.cond(is_training, lambda: tf.nn.dropout(conv2, keep_prob = 0.75), lambda: conv2)

    # Flatten. Input = 5x5x64. Output = 1600.
    fc0 = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 1600. Output = 1024.
    wd1 = tf.Variable(tf.truncated_normal(shape=(1600, 1024), mean = mu, stddev = sigma))
    bd1 = tf.Variable(tf.zeros(1024))
    fc1 = tf.matmul(fc0, wd1) + bd1
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 1024. Output = 512.
    wd2 = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma))
    bd2 = tf.Variable(tf.zeros(512))
    fc2 = tf.matmul(fc1, wd2) + bd2
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.cond(is_training, lambda: tf.nn.dropout(fc2, keep_prob = 0.5), lambda: fc2)

    # Layer 5: Fully Connected. Input = 512. Output = 43.
    wd3 = tf.Variable(tf.truncated_normal(shape=(512, output_size), mean = mu, stddev = sigma))
    bd3 = tf.Variable(tf.zeros(output_size))
    logits = tf.matmul(fc2, wd3) + bd3
    weights = [wd1,wd2,wd3]
    
    return logits, weights


from sklearn.utils import shuffle

save_file = './lenet'
EPOCHS = 100
BATCH_SIZE = 256

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
is_training = tf.placeholder(tf.bool)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits, weights = LeNet(x, is_training)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

# L2 Regularization 
regularization = 0.0
for w in weights:
    regularization += tf.nn.l2_loss(w)

#http://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html
L2_rate = 1e-4

loss_operation = tf.reduce_mean(cross_entropy) + L2_rate*regularization
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, is_training: False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        if i%5 == 0:
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
        
    saver.save(sess, save_file)
    print("Model saved")
