#%% [markdown]
# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
#%% [markdown]
# ---
# ## Step 0: Load The Data

#%%
# Load pickled data
import pickle

# Fill this in based on where you saved the training and testing data
def load_data(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    X, y = data['features'], data['labels']
    return X, y

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'
    
X_train, y_train = load_data(training_file)
X_valid, y_valid = load_data(validation_file)
X_test, y_test = load_data(testing_file)

#%% [markdown]
# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 
#%% [markdown]
# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

#%%
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import pandas as pd
import numpy as np

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

#  How many unique classes/labels there are in the dataset.

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)

df = pd.DataFrame({'CLASS': y_train, 'INDEX': range(0, n_train)})
n_classes = df['CLASS'].nunique()
print("Number of train classes =", n_classes)

#%% [markdown]
# ### Include an exploratory visualization of the dataset
#%% [markdown]
# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

#%%
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# Counts of each class
class_counts = df['CLASS'].value_counts().sort_index()
# print(class_counts)
# y_pos = range(0, len(class_counts))
# plt.bar(y_pos, class_counts, align='center', alpha=0.5)
# plt.xticks(y_pos, y_pos)
# plt.ylabel('Count')
# plt.show()
class_counts.plot(kind='bar')

#%%
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.axis('off')
        a.set_title(title)
        plt.imshow(image)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

import sys
import time

def get_time_hhmmss(start = None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str   

# Print iterations progress
def print_progress(iteration, total):
    """
    Call in a loop to create terminal progress bar
    
    Parameters
    ----------
        
    iteration : 
                Current iteration (Int)
    total     : 
                Total iterations (Int)
    """
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '█' * filled_length + '-' * (100 - filled_length)

    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

#%%
traffic_signs = df.drop_duplicates(subset='CLASS', keep="first")
sorted = traffic_signs.sort_values(by=['CLASS'])
sign_idx = sorted['INDEX']
sign_images = X_train[sign_idx]
titles = y_train[sign_idx]

show_images(sign_images, cols=6)

#%% [markdown]
# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
#%% [markdown]
# ### Pre-process the Data Set (normalization, grayscale, etc.)

#%% [markdown]
# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

#%%
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
import cv2
from skimage import exposure

def normalize(data): 
    # standardisation
    #data = (data-data.mean())/(data.std())

    # convert from [0:255] to [0:1]
    data = (data - data.min()) / (data.max() - data.min())

    # convert from [0:255] to [-1:+1]
    #data = ((data / 255)-0.5)*2

    return data

def equalizeHistYUV(img):
    # Apply Histogram equalization for YUV color space to enhance the contract, and minimize the impact of shadows and lights.
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    gray = img_yuv[:,:,0]
    gray = normalize(gray)
    # equalize the histogram of the Y channel
    # img_output = cv2.equalizeHist(gray)
    # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_output = exposure.equalize_adapthist(gray)
    # show_images([origin_img, gray, img_output], cols=1)
    img_output = img_output.reshape(img_output.shape + (1,))
    return img_output

#%%
pre_process_imgs = np.zeros([sign_images.shape[0], sign_images.shape[1], sign_images.shape[2], 1])
for idx in range(len(sign_images)):
    origin_img = sign_images[idx]
    img_output = equalizeHistYUV(origin_img)
    pre_process_imgs[idx] = img_output
    print_progress(idx+1, len(sign_images))

show_images(pre_process_imgs[:,:,:,0], cols=4)
# show_images([origin_img, img_output[:,:,0]], cols=1)

#%% [markdown]
# ### Augmentation

#%% [markdown]
# The training data amount in initial pickle file is not quite enough to train a model well. It also unbalanced, some classes are have extremly less occurances than others. We could improve this by data augmentation.

#%%
import random

# Flipping
def image_flip(X, y):
    flip_horiz = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    flip_vert = np.array([1, 5, 12, 15, 17])
    flip_both = np.array([32, 40])
    cross_flip = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [39, 38],
        [37, 36],
        [34, 33],
        [20, 19]
    ])
    num_classes = 43
    
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.dtype)
    
    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y == c], axis = 0)
        if c in flip_horiz:
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis = 0)
        if c in cross_flip[:, 0]:
            flip_class = cross_flip[cross_flip[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        
        if c in flip_vert:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        
        if c in flip_both:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    
    return X_extended, y_extended

from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform

# Rotate image randomly
def rotation(X, intensity):
    for i in range(X.shape[0]):
        delta = 30 * intensity
        X[i] = rotate(X[i], random.uniform(-delta, delta), mode='edge')
    return X  
    
def projection(X, intensity):
    image_size = X.shape[1]
    delta = image_size * 0.3 * intensity
    for i in range(X.shape[0]):
        # Top left corner, top margin
        tl_top = random.uniform(-delta, delta)
        
        # Top left corner, left margin
        tl_left = random.uniform(-delta, delta)
        
        # Bottom left corner, bottom margin  
        bl_bottom = random.uniform(-delta, delta)
        
        # Bottom left corner, left margin
        bl_left = random.uniform(-delta, delta)
        
        # Top right corner, top margin
        tr_top = random.uniform(-delta, delta)
        
        # Top right corner, right margin
        tr_right = random.uniform(-delta, delta)
        
        # Bottom right corner, bottom margin
        br_bottom = random.uniform(-delta, delta)
        
        # Bottom right corner, right margin
        br_right = random.uniform(-delta, delta)

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        X[i] = warp(X[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')

    return X

def apply_transform(X, intensity):
    X = rotation(X, intensity)
    X = projection(X, intensity)
    return X 

def extend_data(X, y, intensity, num_classes, output_count=0):
    # We want to have equal amount of data for all classes
    # Get the max count among all classes
    _, class_counts = np.unique(y, return_counts=True)
    max_c = max(class_counts) if output_count==0 else output_count
    total = max_c * num_classes if output_count==0  else output_count

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = np.float32)
    y_extended = np.empty([0], dtype = y.dtype)

    for c, count in zip(range(num_classes), class_counts):
        X_source = (X[y == c]/255.).astype(np.float32)
        y_source = y[y == c]
        X_extended = np.append(X_extended, X_source, axis = 0)
        for i in range((max_c // count) - 1):
            new_x = apply_transform(X_source, intensity)
            X_extended = np.append(X_extended, new_x, axis = 0)
            print_progress(X_extended.shape[0], total)
        
        remain = max_c % count
        new_x = apply_transform(X_source[:remain], intensity)
        X_extended = np.append(X_extended, new_x, axis = 0)
        print_progress(X_extended.shape[0], total)

        # Fill labels for added images set to current class.
        extend_size = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((extend_size), c, dtype = int))

    X_extended = (X_extended*255.).astype(np.uint8)
    return X_extended, y_extended

#%%
train_balanced_data_file = "traffic-signs-data/train_balanced.p"

print("Number of origin data = ", X_train.shape[0])
X_train, y_train = image_flip(X_train, y_train)
print("After flip = ", X_train.shape[0])
X_train, y_train = extend_data(X_train, y_train, 0.8, n_classes)
print("After extended = ", X_train.shape[0])
pickle.dump({
    "features" : X_train,
    "labels" : y_train
}, open(train_balanced_data_file, "wb" ) )

# df_b = pd.DataFrame({'CLASS': balanced_y})
# counts = df_b['CLASS'].value_counts().sort_index()
# counts.plot(kind='bar')

# Test Augmentation
# img1 = sign_images[0]
# x1 = np.full([5, img1.shape[0], img1.shape[1], img1.shape[2]], img1)
# show_images(x1)
# x2 = apply_transform((x1/255.).astype(np.float32), 0.75)
# x2 = (x2*255.).astype(np.uint8)
# show_images(x2)

#%%
def preprocessImages(images):
    out_imgs = np.zeros([images.shape[0], images.shape[1], images.shape[2], 1])
    num = len(images)
    for index in range(num):
        if index%100 == 0:
            print_progress(index+1, num)
        out_imgs[index] = equalizeHistYUV(images[index])

    return out_imgs.astype(np.float32)
#%%
X_train = preprocessImages(X_train)
X_valid = preprocessImages(X_valid)
X_test = preprocessImages(X_test)

#%%
train_balanced_data_file_preprocessed = "traffic-signs-data/train_balanced_preprocessed.p"
test_data_file_preprocessed = "traffic-signs-data/test_preprocessed.p"
valid_data_file_preprocessed = "traffic-signs-data/valid_preprocessed.p"

n_bytes = 2**31
max_bytes = 2**31 - 1

bytes_out = pickle.dumps({
    "features" : X_train,
    "labels" : y_train
})

# Save Large data > 2GB
with open(train_balanced_data_file_preprocessed, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

#%%
X_valid = preprocessImages(X_valid)
pickle.dump({
    "features" : X_valid,
    "labels" : y_valid
}, open(valid_data_file_preprocessed, "wb" ) )

X_test = preprocessImages(X_test)
pickle.dump({
    "features" : X_test,
    "labels" : y_test
}, open(test_data_file_preprocessed, "wb" ) )

#%%
import pickle

import os.path
train_balanced_data_file_preprocessed = "traffic-signs-data/train_balanced_preprocessed.p"

def load_large_data(path):
    bytes_in = bytearray(0)
    input_size = os.path.getsize(path)
    with open(path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    X, y = data['features'], data['labels']
    return X, y

X_train, y_train = load_large_data(train_balanced_data_file_preprocessed)
print(X_train.shape)

#%%
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

#%% [markdown]
# ### Model Architecture
#%%
### Define your architecture here.

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

#%% [markdown]
# ### Train, Validate and Test the Model

#%% [markdown]
# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

#%%
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
from sklearn.utils import shuffle

save_file = './lenet'
EPOCHS = 100
BATCH_SIZE = 128

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

#%%
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
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, save_file)
    print("Model saved")

#%%
from pandas.io.parsers import read_csv

def plot_learning_curve(history):
    e = np.arange(1, history.shape[0], 1)
    a = history[e-1]

    fig, ax = plt.subplots()
    ax.plot(e, a)

    ax.set(xlabel='EPOCH', ylabel='Accuracy',
       title='Learning Rate')
    ax.grid()

    fig.savefig("learning_rate.png")
    plt.show()

history = read_csv("history.csv").values[:, 1]
plot_learning_curve(history)

#%% [markdown]
# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
#%%
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

#%% [markdown]
# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.
#%% [markdown]
# ### Load and Output the Images

#%%
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.image as im
from pandas.io.parsers import read_csv

sign_names = read_csv("signnames.csv").values[:, 1]
X_new = np.empty([0, 32, 32, 3], dtype=np.uint8)
for i in range(8):
    path = 'traffic-signs-data/new/' + "{0:0>3}".format(i) + '.png'
    image = im.imread(path)
    image = (image * 255).round().astype(np.uint8)
    X_new = np.append(X_new, [image[:,:,:3]], axis = 0)

show_images(X_new)
y_new = np.array([34,40,17,39,28,13,28,24])

#%% [markdown]
# ### Predict the Sign Type for Each Image

#%%
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
# Preprocessing the images
X_new_preproccessed = preprocessImages(X_new)
show_images(X_new_preproccessed[:,:,:,0])

#%%
# Predict the Top 5 result 
def prediction(X_input, y_input, k):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    is_training = tf.placeholder(tf.bool)

    logits, _ = LeNet(x, is_training)
    predictions = tf.nn.softmax(logits)
    top_5 = tf.nn.top_k(predictions, k)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        values, indices = sess.run(top_5, feed_dict={x: X_input, y: y_input, is_training: False})
        return values, indices
#%%
ps, idx = prediction(X_new_preproccessed, y_new, k=5)
print(ps)
print(idx)

#%%
correct_count = 0
for i in range(len(X_new)):
    show_images([X_new[i], X_new_preproccessed[i,:,:,0]])
    print("Actual class: ", sign_names[y_new[i]])
    print("\nPrediction: ")
    p = ps[i]
    c = idx[i]
    for j in range(len(p)):
        text = f"Class {c[j]} " + sign_names[c[j]] + f" : {p[j]*100:.2f}%"
        print(text)
    print("---------------------------------------------------------")
    
    if y_new[i]==c[0]:
        correct_count += 1

#%% [markdown]
# ### Analyze Performance
#%%
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
total_num = len(idx)
accuracy = 100.0 * (correct_count/total_num)
print("Accuracy on new images: {:.2f}%".format(accuracy))

#%% [markdown]
# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web
#%% [markdown]
# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

#%%
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

#%% [markdown]
# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
#%% [markdown]
# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
#%% [markdown]
# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

#%%
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


