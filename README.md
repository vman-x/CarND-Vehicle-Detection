**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/windows.jpg
[image4]: ./output_images/1.png
[image5]: ./output_images/2.png
[image6]: ./output_images/3.png
[image7]: ./output_images/4.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/thresholded.png
[image10]: ./output_images/detected.png
[video1]: ./project_video_op.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the sixth code cell of the IPython notebook  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

There are a total  8792  car samples and 8968 non car samples which is pretty much equal as this would avoid one feature dominating over the other.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I mostly experimented with HSV, YUV and LUV color spaces. Among them I got satisfactory results (accuracy and training time) with the LUV color space and following values for the parameters:

|Parameter|Value|
|--|--|
|color_space| LUV |
|HOG Orientation | 8 |
|Pixel per cell|8x8|
|Cell per block|2x2|
|HOG Channel|0|
|Spatial binning dimensions|(16,16)|
|Spatial features|On|
|Histogram features |On|
|HOG features|On|
 
 With these tuned parameters I got an accuracy of 0.9944 and training time of 5.06s with Linear SVC and 193.61 seconds with non linear kernel SVC.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code to train the Classifier can be found in the seventh cell of the python notebook.

Before training the classifier the extracted data was normalized using the `StandardScaler()` function which reduces the mean to zero and variance to one.
I trained non-linear SVC with `rbf` kernel using `svc.fit()` function. I fed the fit function with preprocessed features and labels extracted from the previous step. I varied the classifiers between Linear and non-linear and found out non-linear classifier provided a better accuracy.

Accuracy was calculated using the `score` function.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I first did a little research to come up with a ROI within which the car can be found. For that I set the y start position to 400 so that the area above the horizon is ignored while searching for the object. 

Then I tried with different window sizes by measuring the size of the car at different y positions.
(i.e.) Cars near by to the car appear larger hence a large window size, say `128x128px`. Similarly cars near to the horizon have a smaller size hence a window size of `64x64px`.
Also I added a intermediate window size of `96x96px` so that it can detect cars in between the ranges.

For the overlapping percentage, I actually experimented with different percentages varying from 40% to 80% for both x and y axis. Finally I got 50% in x axis and 75% in y axis proved to be pretty accurate while keeping the number of windows to search - small value.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Finally I used a `LUV` color space with HOG channel of 0. Also I combined spatial binning and color histogram along with it to get really good results. 
 
Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_op.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detection in each frame of the video.  From the positive detection I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.   
Below you can find the different stages of processing the frame:

First stage
![alt text][image3]

Heat map
![alt text][image8]

Thresholding & labelling
![alt text][image9]

Final detection
![alt text][image10]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At first I tried to complete the project without using a non-linear classifier to reduce the cost of training the classifier. However I was not very successful in implementing the pipeline using linear classifier. Then I switched to non linear classifier and it gave me good predictions at the increased cost of training.

I would like to implement the model using a linear classifier since it is very optimal in real case scenario. As a demerit of using a non linear classifier, the time to process each frame increases exponentially. It is evident from the fact that linear classifier took only 6 mins to process 50 second video whereas non linear classifier took more than 30 mins to process the same.

According to me the model would fail where decisions to be taken within split second and the pipeline would be taking several minutes to process the data. It is not ideal. As I said, I would try to implement the same in Linear classifier over this weekend.

> Best Regards
> Vivek Mano
