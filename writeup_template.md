# **Traffic Sign Recognition** 

The aim of the project is to build a model that can classify traffic signs with a validation accuracy equal to 0.93 or greater.
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is a link to my [project code](https://github.com/meenunat/German_traffic_Sign_classifier/blob/master/Traffic_Sign_classifier.ipynb)

## Data Set Summary & Exploration

[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) was used for training/ Validation/ Testing the model.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

## Include an exploratory visualization of the dataset.

From the data distribution, it is obvious that data set is highly skewed.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![](results/data_distribution.png?raw=true "German Traffic Signal Data Distribution")

![](results/data_0.png?raw=true "Speed limit (20km/h)")

![](results/data_1.png?raw=true "Speed limit (30km/h)")

![](results/data_2.png?raw=true "Speed limit (50km/h)")

## Design and Test a Model Architecture

The original LeNet Model was chosen initially with 3 fully connected  layers. The validation accuracy and training accuracy indicated the model was over-fitting with high training accuracy(0.987) and low validation accuracy(.81). To improve the accuracy, an additional fully connected layer was added to the LeNet Model. The Model has 2 convolution layers which is followed by pooling and a relu activation layer with 50% data dropout. The output of the model is processed via four fully connected layers after flattening.

### Preprocessing

The original RGB image was converted into different color space ['Gray','HSV', 'HLS', 'Lab', 'Luv', 'XYZ', 'Yrb', 'YUV'] before parsing it through the model. Conversion to gray scale and yuv space showed increased validation accuracy than other conversion techniques (0.888).

### Original Lenet

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x1x6	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			     	|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU			     	|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				|         					     				|
| Fully connected		| Input: 400 and Output:120   					|
| Fully connected		| Input: 120 and Output:84   					|
| Fully connected		| Input: 84 and Output:10   					|


### Model build in the project:
Due to over-fitting of data, an additional fully connected layer was added to the original LeNet Model. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution			| Filter is choosen based on image - gray/color |
|	5x5x1x32			| 1x1 stride, valid padding, outputs 28x28x32 	|
|	5x5x3x32			| 1x1 stride, valid padding, outputs 28x28x32	|
| RELU			     	|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32					|
| Convolution 5x5x32x64 | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU			     	|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Dropout	      		| Probability - 50% 							|
| Flatten				|         					     				|
| Fully connected		| Input: 1600 and Output:540   					|
| Fully connected		| Input: 540 and Output:180   					|
| Fully connected		| Input: 180 and Output:86   					|
| Fully connected		| Input: 86 and Output:43   					|

Normalization of the data showed an increase in the validation accuracy (0.916) and training accuracy(0.996). But the model exposes an oscillation in the accuracy. 

In order to overcome the oscillation, learning rate was decreased by 10. But the accuracy decreased. To help this, model needs to be generalized for which dropout of 50% was added after second convolution layer and the number of epochs were also increased.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.946
* test set accuracy of 0.939


* What was the first architecture that was tried and why was it chosen? 
LeNet Architecture was chosen with 3 fully connected layers.

* What were some problems with the initial architecture?
The model was overfitting the data with the initial architecture. 
 
* How was the architecture adjusted and why was it adjusted? 
In order to improve this, an additional layer was added to the original LeNet model with 50% dropout to improve generalization.  

* Which parameters were tuned? How were they adjusted and why?
Learning rate, number of epochs and batch size were tuned to get a robust model.

## Test a Model on New Images
The model is designed to accept only images of size 32x32, While the web images were of bigger size. Scaling the images may tend to loose information leading to wrong prediction.

### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

German traffic signs that I found on the web:

![](results/webimage.png?raw=true "Webimages")

The images might be difficult to classify because of varied brightness in the image. 

![](results/webimage.png?raw=true "Webimages")

![](results/webimage_result_1.png?raw=true "Wrong Prediction 1")

![](results/webimage_result_5.png?raw=true "Wrong Prediction 2")

Also, scaling the image may have caused crucial information to be lost.

![](results/webimage_result_3.png?raw=true "Wrong Prediction 3")

![](results/webimage_result_4.png?raw=true "Wrong Prediction 4")

The model was able to correctly guess 3 out of the 7 traffic signs, which gives an accuracy of 42.8%. This triggers an interesting discussion to augment data to have a balanced datasets which might potentially help in better prediction accuracy.

In real world, environment, the requirement is to have faster model with better prediction. The current model takes a longer time to process due to low learning rate which is not desirable. 

## Additional Improvements:
1.	Need to augment data to get a balanced dataset
2.	The model needs to be modified to work on videos