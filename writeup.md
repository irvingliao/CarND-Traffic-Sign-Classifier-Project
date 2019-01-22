# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/hist.png "Visualization"
[image2]: ./Images/Preprocess_before.png "Before Hist Equalization"
[image3]: ./Images/Preprocess_after.png "After Hist Equalization"
[image4]: ./Images/flip_h1.png "Flip horizontal 1"
[image5]: ./Images/flip_h2.png "Flip horizontal 2"
[image6]: ./Images/flip_v1.png "Flip vertical 1"
[image7]: ./Images/flip_v2.png "Flip vertical 2"
[image8]: ./Images/flip_x.png "Flip Mix"
[image9]: ./Images/flip_hx1.png "Flip horizontal change 1"
[image10]: ./Images/flip_hx2.png "Flip horizontal change 2"
[image11]: ./Images/warp_before.png "Warp before"
[image12]: ./Images/warp_after.png "Warp after"
[image13]: ./Images/learning_curve.png "Learning Curve"
[image14]: ./Images/Test_origin.png "Test Images Origin"
[image15]: ./Images/Test_preprocessed.png "Test Images Preprocessed"
[image16]: ./Images/Test_1.png "Test Image 1"
[image17]: ./Images/Test_2.png "Test Image 2"
[image18]: ./Images/Test_3.png "Test Image 3"
[image19]: ./Images/Test_4.png "Test Image 4"
[image20]: ./Images/Test_5.png "Test Image 5"
[image21]: ./Images/Test_6.png "Test Image 6"
[image22]: ./Images/Test_7.png "Test Image 7"
[image23]: ./Images/Test_8.png "Test Image 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/irvingliao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I'm using numpy and pandas method to analytics the data by following code:
```
n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
```
```
df = pd.DataFrame({'CLASS': y_train, 'INDEX': range(0, n_train)})
n_classes = df['CLASS'].nunique()
```

Here's the data we get:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing the training data classification in all 43 signs.
I notice that the number of training images in each sign is not equally distributed.
Like Class 1 Sign is about 2000, but Class 0 is only about 200 counts.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here's the steps that I desing the Preprocessing procesure.

##### **Apply Histogram equalization for YUV color space to enhance the contract, and minimize the impact of shadows and lights**
```
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
gray = img_yuv[:,:,0]
gray = normalize(gray)
img_output = exposure.equalize_adapthist(gray)
img_output = img_output.reshape(img_output.shape + (1,))
return img_output
def equalizeHistYUV(img):    
```
Here is an example of a traffic sign image before and after Histogram equalization.
![alt text][image2]
![alt text][image3]

We notice it can minimize the effect of shadows and light, and get the sign outlines.

##### Normalize the data
We would like to convert the pixel range [0...250] to [0...1].
Because it can reduce the compute time for image processing and also can minimize the precision loss.

##### Data Augmentation
The training data amount in initial pickle file is not quite enough to train a model well. It also unbalanced, some classes are have extremly less occurances than others. We could improve this by data augmentation.

To add more data to the the data set, I used the following techniques:
Flip

Here is an example of an original image and an augmented image:
##### Flip
A simple trick to extend the data by simply flip the image in axis.

We can flip horizontally:

![alt text][image4] ---> ![alt text][image5]

Flip Vertically:

![alt text][image6] ---> ![alt text][image7]

Flip in both ways:

![alt text][image8]

In some case, after flipping, it will be recognized as another class

![alt text][image9] ---> ![alt text][image10]

```
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
```

##### Rotate & Warp
To extend all data equally, for thos classes which have less traning data originally, we can apply a random rotation and warp to augment the dataset.

Apple Random Rotation & Warp transform to the original image.
![alt text][image11]
![alt text][image12]

```
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
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model consisted of the following layers, same as LeNet:
2 Convolution layer as feature extraction
3 Fully connected layer as classifier

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x32, dropoup prob 0.9 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 10x10x64, dropoup prob 0.75 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 		     		|
| Flatten	         	| outputs 1600              		     		|
| Fully connected		| output 1024       							|
| Fully connected		| output 512   dropoup prob 0.5     			|
| Fully connected		| output 43         							|
|						|												|

```
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
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### Regulation
There's 2 main thing I'm using to minimize the overfitting to training dataset.

###### Dropout
| Layer         		|  Dropout  	|  Keep Prob     |
|:---------------------:|:-------------:|:--------------:|
| Conv 5x5 1      		|  10%          | 0.9	         |
| Conv 5x5 2      		|  25%          | 0.7	         |
| Fully connected 3		|  50%          | 0.5	         |

I applied Dropout to improve the generalization of the training model.
By increasing the drop probabiliity gradually in deeper layer, it did perform better.

###### L2 Regularization
I apply 0.0001 as L2 rate, which seem to perform better. L2 loss should only include weights of the fully connected layers, and we won't include bias, becuase bias is not contribute overfitting.

Then I set **EPOCHS = 100**, **BATCH_SIZE = 256**, **Learning rate = 0.001**, and AdamOptimizer.
After 60 Epoch, the model accuracy is about the **98.2%**, and didn't change a lot.
At the stop of Epoch 100, the accuracy is reaching **98.9%**

```
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
```

Here's the learning curve by epoch:

![alt text][image13]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy: 98.9% 
* test set accuracy of: 96.9%

If a well known architecture was chosen:
* What architecture was chosen?
I'm using the LeNet which introduced in course.
* Why did you believe it would be relevant to the traffic sign application?
The convolutional layer to fetch features and fully connected layer to classification seem to be pretty well perform.
By apply seme regularization technique, it improve well enough
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
I didn't run the full Epoch every time, I usually try to slice a random small dataset to train the model and see how it initially looks like.
If the architecture or the value is working good enough, usually it will reach over 95% in first 3 epoch.

I think if I turn down the turning rate to 0.0001 or 0.00001 and using anther randomly generated augmented dataset to continue another round of training, it would improve a bit better.

However, I don't have any GPU computer, I was using my macbook pro to run training on CPU only over 3 more hours to finish all 100 epoch.

Simply have no time to do too much run of training.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image14]

After Preprocessing

![alt text][image15]

The #5 sign, Children crossing, might be diffical to predict, because it is not the same Children crossing sign used in training, and it is a little blur.
It might hard to see it is 2 people in the sign or 1 signal pattern.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

99.9% as Turn left ahead, which is pretty correct.
![alt text][image16]

100% as Roundabout mandatory
![alt text][image17]

100% as No Entry
![alt text][image18]

99.6% as keep left
![alt text][image19]

98.6% as Bicycles Crossing,
1.36% as Children Crossing
We can see the prediction of this is more to be Bicycles Crossing, which is reasonable, the sign is pretty similar to Children Crossing in blur.
![alt text][image20]

100% as Yield
![alt text][image21]

79.35% as Children Crossing,
19.84% as Bicycles Crossing
This image has more contract in image for feature of 2 people pattern, which lead it more lean to Children Crossing than Bicycle Crossing.
![alt text][image22]

94.81% as Road narrow on the right
![alt text][image23]

The prediction of the test images are correct for 7 of 8, the accuracy is 87.5% for this test data set.
