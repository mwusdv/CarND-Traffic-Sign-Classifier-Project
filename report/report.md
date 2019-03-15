# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

**Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.**

### Writeup / README
**1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.**

You're reading it! and here is a link to my [project code](https://github.com/mwusdv/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration
**1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is ?  34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

**2. Include an exploratory visualization of the dataset.**

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
![class bar chart](class_bar_chart.jpg)


### Design and Test a Model Architecture
**1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)**

* **Histogram equalization.** The first step of pre-processing is histogram equalization. This is to reduce the impact on the different intensity distributions in each image.  Here is an example of a traffic sign image before and after histogram equalization. We can see that the two original images have quite different intensity distributions. And this difference is intuitively reduced after histogram equalization.

    ![histeq](pre-process.jpg)

* **Normalization.** As the second step of pre-processing: each image is normlaized by subtracting the mean and dividing by the standard deviation of that image. This can make the each image have the mean value of 0.0 and standard deviation of 1.0. By making the distribution of the images similar to each other, the training and generalization could be easier.

* **Generating more training examples.** Next I decided to generate additional data because as can be seen from the above bar chart, the number of training examples are highly imbalanced. And also the size, position, and angles vary a lot within the same classs. Therefore, 
    
    * To make the number of training examples equal to each other in different classes.

    * I used `cv2.warpAffine` to generate augmented images. Namely, each image is transformed by rotating, translation, scaling and shearing. The rotation angle, translation amount, scaling amount and shearing parameters are randomly selected within pre-defined ranges. Here is an example of an original image and an augmented image. The first and the 3rd rows are the images from the original training data. While the 2nd and the 4th rows are the augmented images. It can be seen that the augmented images have rotaions, translations, scaling and shearing.

        ![aug_img](aug.jpg)
    
   
2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

    My final model architecture can be best described by the parameters in my code. This format gives me much flexibility to implement the network. I was able to experiment different network architectures by just chaning the parmaeters withoug changing the code at all.

    ```
    # pre-processing layers
    pre_prop_layers = [{'kernel': [3, 8], 'pooling': True, 
    'keep_prob': 1.0, 'go_to_fc': False, 
    'activation_fn': tf.nn.relu, 'padding': 'SAME', 
    'batch_norm': True, 'l2_reg': 0.01},
                            
    {'kernel': [1, 8], 'pooling': True, 
    'keep_prob': 0.8, ' go_to_fc': False, 
    'activation_fn': tf.nn.relu, 'padding': 'SAME', 
    'batch_norm': True, 'l2_reg': 0.01}]



    # conv layers: 
    conv_layers = [{'kernel': [[3, 16], [5, 16], [7, 3, 16], [3, 7, 16]], 'pooling': True, 'keep_prob': 0.8, 
    'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True, 'l2_reg': 0.01},
        
    {'kernel': [[3, 32], [5, 32], [7, 3, 32], [3, 7, 32]], 
    'pooling': True, 'keep_prob': 1.0, 
    'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True, 'l2_reg': 0.01},

    {'kernel': [[3, 64], [5, 2, 64], [2, 5, 64]], 
    'pooling': True, 'keep_prob': 1.0,       
    'go_to_fc': True, 'activation_fn': tf.nn.relu, 'padding': 'SAME', 'batch_norm': True, 'l2_reg': 0.01}]



    # fully connected layers
    self._fc_layers = [{'hidden_dim': 512, 'keep_prob': 0.5, 
    'activation_fn': tf.nn.relu, 'batch_norm': True, 'l2_reg': 0.1}]
    ```
    In the above code block, the meaning of most of the parameters are quite straight forward. The meaning of the _kernel_ parameters will be described in the following.

    There are three blocks of layers in my model. 
    * The first block consists of **pre-processing** layers. I didn't do grayscaling in the previous pre-processing step. Rather I choosed to do the _pixel combinations_ here and let the algorithm to choose the optimal weights.  In this block, the _kernel_ parameters consists of two integer values. The first one is the size of the convolutional kernel, while the second one is the number of kernels.

    * The second block is made up of 3 convolutional layers
Input 	32x32x3 RGB image
Convolution 3x3 	1x1 stride, same padding, outputs 32x32x64
RELU 	
Max pooling 	2x2 stride, outputs 16x16x64
Convolution 3x3 	etc.
Fully connected 	etc.
Softmax 	etc.
	
	
3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....
4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

    training set accuracy of ?
    validation set accuracy of ?
    test set accuracy of ?

If an iterative approach was chosen:

    What was the first architecture that was tried and why was it chosen?
    What were some problems with the initial architecture?
    How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    Which parameters were tuned? How were they adjusted and why?
    What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:

    What architecture was chosen?
    Why did you believe it would be relevant to the traffic sign application?
    How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Test a Model on New Images
1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

alt text alt text alt text alt text alt text

The first image might be difficult to classify because ...
2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
Image 	Prediction
Stop Sign 	Stop sign
U-turn 	U-turn
Yield 	Yield
100 km/h 	Bumpy Road
Slippery Road 	Slippery Road

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...
3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
Probability 	Prediction
.60 	Stop sign
.20 	U-turn
.05 	Yield
.04 	Bumpy Road
.01 	Slippery Road

For the second image ...
(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?