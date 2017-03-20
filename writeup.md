**Traffic Sign Recognition**

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

[image1]: ./examples/distribution.png "Traffic Sign Class Distribution"
[image2]: ./examples/animal-before.png "Before Processing"
[image3]: ./examples/animal-after.png "After Processing"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
**Writeup / README**

You're reading it! and here is a link to my [project code](https://github.com/spillow/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

***1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.***

The code for this step is contained in cell 4.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

***2. Include an exploratory visualization of the dataset and identify where the code is in your code file.***

The code for this step is contained in cells 5-6.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the traffic signs in the training data.
Note that there is a spread of roughly 10:1 between the most commonly observed signs and the least common.

![alt text][image1]

**Design and Test a Model Architecture**

***1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.***

The code for this step is contained in cells 7-8 of the IPython notebook.

Transforming the images to grayscale was first tried to see if eliminating color information had much effect on validation accuracy.  With the LeNet architecture,
it didn't vary significantly.

It looked like histogram equalization techniques would help in removing brightness variance that would make it more difficult to train.  Specifically, OpenCV's
CLAHE (Contrast Limited Adaptive Histogram Equalization) worked out well after experimeting with block sizes.  Here is an example before and after transformation (+ image normalization):


![alt text][image2] ![alt text][image3]

Normalization of the data (zeroing out the mean and diving by the standard deviation for each pixel) was considered but, since these features are pixel values, there
shouldn't be any great difference in their scale.  See [this](http://cs231n.github.io/neural-networks-2/) for more discussion on the matter.

***2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)***

The train/validation/test split was already provided as a set of pickled blobs and are loaded in cell 1.  Data breakdown:

| Type         | Count
|:------------:|--------:|
| Training     | 34799   |
| Validation   | 4410    |
| Test         | 12630   |

One method to achieve this breakdown is to shuffle a collection of data and then assign a percentage to each category.  An 80/10/10 split is common but, like here, the majority should reside
in training.

***3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.***

The architecture template is in cell 10.  I refactored the initial LeNet model code to take as arguments a list of convolution layer and fully connected layer dimensions to aid in quickly
prototyping different architectures.

The final model consists of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							            |
| Convolution 1x1x3   	| 1x1 stride, valid padding, outputs 32x32x3 	  |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					        |												                        |
| Convolution 15x15    	| 1x1 stride, valid padding, outputs 14x14x40 	|
| RELU					        |												                        |
| Convolution 4x4    	  | 1x1 stride, valid padding, outputs 10x10x60 	|
| RELU					        |												                        |
| Convolution 4x4    	  | 1x1 stride, valid padding, outputs 7x7x80 	  |
| RELU					        |												                        |
| Fully connected		    | outputs 300        									          |
| RELU					        |												                        |
| Fully connected		    | outputs 200                                   |
| RELU					        |												                        |
| Fully connected		    | outputs 150                                   |
| RELU					        |												                        |
| Fully connected		    | outputs 120                                   |
| RELU					        |												                        |
| Fully connected		    | outputs 84                                    |
| RELU					        |												                        |
| Fully connected		    | outputs 43                                    |
| Softmax				        | outputs probabilites to cross entropy         |

Notably, after initial experiments max pooling layers were dropped.  There's certainly more room for experimentation here but no overfitting was observed
so there may be an opportunity for more shrinking convolutional layers instead of max pooling.


***4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.***

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
