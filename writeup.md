# **Traffic Sign Recognition**

### [Jupyter Notebook (HTML)](P3.html)

[//]: # (Image References)

[image1]:  ./res/output_images/1_random_images.jpg       "Randomly-generated images"
[image2]:  ./res/output_images/2_label_frequency.jpg     "Label Frequency Histogram"
[image3]:  ./res/output_images/3_grayscaled.jpg          "Grayscaled Image"
[image4]:  ./res/output_images/4_normalized.jpg          "Normalized Image"
[image5]:  ./res/output_images/5_validation_accuracy.jpg "Validation Accuracy"
[image6]:  ./res/output_images/6_training_by_epoch.png   "Validation by Epochs"
[image7]:  ./res/output_images/7_german_signs.jpg        "German Signs"
[image8]:  ./res/output_images/8_softmax_viz.jpg         "Softmax Visualized"
[image9]:  ./res/output_images/9_sign_predictions.png    "Predicted v Actual"
[image10]: ./res/lenet_1.png                             "LeNet Architecture"
[image11]: ./res/lenet_2.jpeg                            "LeNet Architecture"


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a grouping of 15 randomly-generated test images with their associated training labels:

![alt text][image1]

and a histogram of sign training label frequency:

![alt text][image2]

with all label names viewable below and in the following [attached file](./data/signnames.csv).

|   ClassId        		|     SignName  	        					    |
|:---------------------:|:--------------------------------------------------|
| 0              		| Speed limit (20km/h)							    |
| 1                  	| Speed limit (30km/h)                        	    |
| 2 					| Speed limit (50km/h)	      					    |
| 3         	      	| Speed limit (60km/h)          				    |
| 4             	    | Speed limit (70km/h)							    |
| 5             		| Speed limit (80km/h)							    |
| 6     				| End of speed limit (80km/h)					    |
| 7						| Speed limit (100km/h) 						    |
| 8						| Speed limit (120km/h)							    |
| 9						| No passing									    |
| 10					| No passing for vehicles over 3.5 metric tons	    |
| 11					| Right-of-way at the next intersection			    |
| 12    				| Priority road									    |
| 13					| Yield											    |
| 14        			| Stop											    |
| 15					| No vehicles									    |
| 16					| Vehicles over 3.5 metric tons prohibited		    |
| 17    				| No entry										    |
| 18					| General caution								    |
| 19					| Dangerous curve to the left					    |
| 20					| Dangerous curve to the right					    |
| 21					| Double curve									    |
| 22					| Bumpy road									    |
| 23					| Slippery road									    |
| 24					| Road narrows on the right						    |
| 25					| Road work										    |
| 26					| Traffic signals								    |
| 27					| Pedestrians									    |
| 28					| Children crossing								    |
| 29					| Bicycles crossing								    |
| 30					| Beware of ice/snow							    |
| 31					| Wild animals crossing							    |
| 32					| End of all speed and passing limits			    |
| 33					| Turn right ahead							        |
| 34					| Turn left ahead								    |
| 35					| Ahead only									    |
| 36					| Go straight or right							    |
| 37					| Go straight or left							    |
| 38					| Keep right									    |
| 39					| Keep left										    |
| 40					| Roundabout mandatory							    |
| 41					| End of no passing								    |
| 42					| End of no passing by vehicles over 3.5 metric tons|

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because it worked well for Sermanet and LeCun as described in their journal article. The smaller array sizes had an added benefit of also reducing training time.

Here is an example of several traffic sign images before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data mainly because I read that having a wider distribution in the data makes it more difficult to train using a singular learning rate.  In that case, different features could encompass inflated ranges and a single learning rate might make some weights diverge.  In short, it was suggested in the lectures and it was trivial to do.

Here is an example of an original image and an augmented image:

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture was based on the original LeNet Architecture as shown below:

![alt text][image10]

This final model that I designed was a modified version of the architecture the LeNet Lab, with no changes since the input dataset is in grayscale.  This model worked well with a ~92% validation accuracy.  The modifications were inspired by the Sermanet and LeCun model from their traffic sign classifier journal article and is shown below:

![alt text][image11]

The paper doesn't go into detail describing the depth of the layers, so I did some additional research to bring it together.  This final model showed an improved validation accuracy of ~97% and consisted of the following layers:

| Layer         		|     Description	        					        |
|:----------------------|:------------------------------------------------------|
| Input (preprocessing)	| 32x32x3 in, 32x32x1 out					            |
| 5x5 Convolution      	| 32x32x1 in, 1x1 stride, 28x28x6 out 	                |
| ReLU					|												        |
| 2x2 Max pooling      	| 28x28x6 in, 2x2 stride, 14x14x6 out			        |
| 5x5 Convolution	    | 14x14x6 in, 1x1 stride, 10x10x16 out			        |
| ReLU					|												        |
| 2x2 Max pooling      	| 10x10x16 in, 2x2 stride, 5x5x16 out			        |
| 5x5 Convolution	    | 5x5x16 in, 1x1 stride, 1x1x400 out			        |
| ReLU					|												        |
| Flatten Layers 2 & 3  | (5x5x16=400) + (1x1x400=400), 800 out                 |
| Dropout               | `keep_prob = max( ( 0.60 - ( EPOCH / 100 ), 0.40 ) )` |
| Fully connected		| 800 in, 120 out								        |
| ReLU					|												        |
| Dropout               | `keep_prob = max( ( 0.60 - ( EPOCH / 100 ), 0.40 ) )` |
| Fully connected		| 120 in, 43 out    							        |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* optimizer: Adam optimizer (LeNet lab)
* batch size: 128
* epochs: 101
* learning rate: 0.00069
* mu: 0
* sigma: 0.1
* dropout keep probability: `keep_prob = max( ( 0.60 - ( EPOCH / 100 ), 0.40 ) )`


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My approach was very much a trial-and-error approach. Guesses were not totally random thanks to experience from the lectures and Tensorflow/LeNet labs.  A log of trialed hyperparameters is shown below and their impact on validation accuracy:

| Attempt   | Description                                            | Accuracy |
|:---------:|:-------------------------------------------------------|:--------:|
| 1	        | epochs: 45, batch: 128, rate: 0.0008, keep_prob: 0.5   | ~94%     |
| 2	        | epochs: 50, batch: 128, rate: 0.0008, keep_prob: 0.5   | ~92%     |
| 3	        | epochs: 50, batch: 128, rate: 0.00085, keep_prob: 0.5  | ~94%     |
| 4	        | epochs: 50, batch: 256, rate: 0.0009, keep_prob: 0.6   | ~95%     |
| ---       | v-- _ADDED 1 FULLY-CONNECTED LAYER WITH 1 DROPOUT_ --v | ---%     |
| 5	        | epochs: 50, batch: 256, rate: 0.0009, keep_prob: 0.6   | ~96%     |
| 6	        | epochs: 50, batch: 128, rate: 0.0009, keep_prob: 0.6   | ~95%     |
| 7	        | epochs: 50, batch: 128, rate: 0.009, keep_prob: 0.6    | ~91%     |
| 8	        | epochs: 50, batch: 128, rate: 0.0005, keep_prob: 0.5   | ~95%     |
| 9	        | epochs: 50, batch: 128, rate: 0.0005, keep_prob: 0.5   | ~95%     |
| 10        | epochs: 50, batch: 128, rate: 0.0007, keep_prob: 0.5   | ~96%     |
| 11        | epochs: 50, batch: 64, rate: 0.0007, keep_prob: 0.5    | ~95%     |
| 12        | epochs: 50, batch: 256, rate: 0.0007, keep_prob: 0.5   | ~95%     |
| 13        | epochs: 50, batch: 64, rate: 0.0004, keep_prob: 0.5    | ~96%     |
| 14        | epochs: 50, batch: 64, rate: 0.0004, keep_prob: 0.5    | ~95%     |
| 15        | epochs: 50, batch: 64, rate: 0.0005, keep_prob: 0.25   | ~96%     |
| 16        | epochs: 50, batch: 32, rate: 0.0005, keep_prob: var    | ~97.3%   |
| 17        | epochs: 75, batch: 128, rate: 0.0007, keep_prob: var   | ~97.1%   |
| 18        | epochs: 101, batch: 128, rate: 0.0009, keep_prob: var  | ~97.5%   |
| 19, final | epochs: 101, batch: 128, rate: 0.00069, keep_prob: var | ~97.6%   |


My final model results were:
* validation set accuracy of 97.0%
* test set accuracy of 95.3%

![alt text][image6]

A chart of validation accuracy by epoch is shown below:

![alt text][image5]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? _I originally chose the LeNet architecture from the lecture notes.  It is shown in the project notes as the function_ `lenet_0( )`.
* What were some problems with the initial architecture? _The validation accuracy was roughly 92% and, though good, I was hoping to improve upon it._
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. _I added a convolutional layer and simplified the fully-connected layers._
* Which parameters were tuned? How were they adjusted and why? _I chose to expand the_ `EPOCHS` to `101` _for as it did not seem like the model was overfitting.  I reduced the_ `learn_rate` to`0.00069` _and introduced a variable_ `keep_prob` _for the dropout layers._
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? _Concatenating the added convolutional layer with the second layer showed a noticeable increase in validation accuracy._

If a well known architecture was chosen:
* What architecture was chosen? _Modified LeNet_
* Why did you believe it would be relevant to the traffic sign application? [_Traffic Sign Recognition with Multi-Scale Convolutional Networks, [Sermanet, LeCun, 2011]_](https://www.researchgate.net/profile/Yann_Lecun/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks/links/0912f50f9e763201ab000000/Traffic-sign-recognition-with-multi-scale-Convolutional-Networks.pdf)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? _Final validation set accuracy of 97.0% and test set accuracy of 95.3%_


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image7]

The images should not be difficult to classify because all are clearly lit and visibly free of damage.  It is possible that these images might occupy a different range in the color space, possibly a range that the model was not trained on.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9]

The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. This does not compare favorably to the accuracy on the test set of 95% and I believe that these test images may have had a coloration mismatch.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the iPython notebook.

For the first image, the model is unsure sure that this is a "keep left" sign and instead calls it a "keep right" with high probability of 0.993, I believe that this is because the model associated arrows with these two signs, but I do not know why "keep left" was not listed as a second option.

The next four images are accurately classified with 100% probability.

The sixth image ("stop") is incorrectly classified as "speed limit (60km/h)" with probability of 0.553.  "Stop" is listed next with probability 0.439 indicating that the model almost made an accurate prediction.

Three of the final four images are accurately classified with 100% probability with the "road work" image being misclassified as "general caution" with a high probability of 0.985.  Since "road work" is not listed as an alternative prediction and because the probability is so high, I believe that the model's performance is suffering from the same issue as the "keep left" sign, namely that the model trained overfit to internal sign shapes.  

![alt text][image8]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I did not attempt this section.
