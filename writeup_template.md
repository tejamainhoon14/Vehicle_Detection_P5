**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Example_of_Training_Set.png
[image2]: ./output_images/HOG_Transform_Example.png
[image3]: ./output_images/Normalization_of_Features.png
[image4]: ./output_images/Model_Accuracy.PNG
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. HOG Feature Extraction

The code for this step is contained in the third and fourth code cell of the VD_Pipeline jupyter notebook. In the second code cell, I have code to load all the GIT and KITTY dataset to train the classifier. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of HOG transform for a random car image selected from training data with parameters as listed below:
`orientations = 10`
`pixels_per_cell=(8, 8)`
`cells_per_block=(2, 2)`
`color_space = 'YCrCb'`
`hog_channel = 'ALL'`

![alt text][image2]

#### 2. HOG Parameter Selection

I tried various combination of parameters by comparing visually as shown in figure above and then later using the SVM classifier accuracy to adjust parameters to increase the model accuracy.

#### 3. Classifier Model (Linear SVM)

I trained a linear SVM using an array of concatenated features extracted as HOG Features, spatial color information and color histogram on total approximately 14,000 training images of cars and non-cars from KITTI and GTI image database combined, cars images being 8792 and non_car images being 8968. Total number of features extracted using all three methods was 8460. The code for this part is in cell 5 of the jypyter notebook. During this training, I observed how changing the image colorscale from RGB to YCrCb improved the training accuracy. With training accuracies varying in the range of 98.8 - 99.8 %, I did not tweak the network training parameters any further. Before training the network, the feature vectors were normalized so that one particular set of features do not dominate the rest. An exampe of feature normalization is shown below:

![alt text][image3]

Here is an example of output after training the model with accuracy and time to train the model and 10 random predictions:

![alt text][image4]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

