# CarND_Project05VehicleDetectionAndTracking
Solution for Udacity Self driving car nano degree fifth project: Vehicle Detection and Tracking

---

**Detection of Cars in a Video Stream**

[//]: # (Image References)

[LCS]: https://i.pinimg.com/originals/63/c8/9a/63c89aba0ed994edcfce462b2a4b2b6b.jpg

[CarHistImg]: ./ReportImages/Histogram/CarExample.png
[CarHistEx]: ./ReportImages/Histogram/CarExampleHist.png
[NonCarHistImg]: ./ReportImages/Histogram/NonCarExample.png
[NonCarHistEx]: ./ReportImages/Histogram/NonCarExampleHist.png

[CarSBImg]: ./ReportImages/SpatialBinning/CarImagee.png
[CarSBEx]: ./ReportImages/SpatialBinning/CarSB.png
[NonCarSBImg]: ./ReportImages/SpatialBinning/NonCarImagee.png
[NonCarSBEx]: ./ReportImages/SpatialBinning/NonCarSB.png

---

### Reflection

Udacity Self Driving Car Nanodegree fifth project is to correctly detect Cars in a video using computer vision and machine learning.

Before discussing the solution to achieve this goal, it is important to understand the problem and how to actually achieve the needed output.

This is a Binary classification problem (i.e. Car vs Non Car), Detection of cars can be multiclass to handle some special cases as partially occluded cars for example.

Cars have some distinct features that can be used to classify them:
  * They appear only in certain segments on the image --> mainly bottom half of the image
  * Their colors are usually high in saturation (not a very robust feature because old cars colors lose saturation over time)
  * They have a clear and consistent shape

Knowing this then it is clear that the tools that can be used are:
  * Spatial filtering
  * Color histograms
  * Edge detection

---

Background Info on file structure:

Going Into this project I knew it would be the largest of the 5 projects, I tried to have all code in one note book but it soon grow out of control and started to become a mess, I have seperated the code to 4 Note books each with a specific purpose that will be discssed below.

---

I have decided, with help of the lessons instructions, to divide the problem to the following steps:
  
  * Visualize examples of the Dataset: (DataVisualization.ipynb)
    Helps to get some sense about the feature spaces and which features can be used to classify the cars correctly.
  * Feature Extraction and Classifier training: (DataExtractionAndClassifierTraining.ipynb)
    Try to extract different features and feed them to a classifier to obtain most accurate classifier
  * Object Detection for both Test images and videos: (TestClassifierAndVideoDetections.ipynb)
    Use the trained classifier to detect the object and apply different methods to eliminate false positives.

---

## First Step: Data Visualization

Udacity provide an training data set that is already scaled to 64*64 pixels, I have decided to use this dataset although it is quite good to note that visually inspecting the data set it seems to have a lot more dark colored images than light colord ones so it would be a better choice to supplement this using other images from other data sets as the Kitti dataset for example, also in the course they mentioned that the GTI images are extracted from a time series so there will be a lot of images with very similar if not identical features so it would make sense to manually divide them to ensure large variation between your training and verficitaion data set to prevent classifier overfitting.

We have three main fetures to work with:

### 1) Color Histogram:

This is by all means a soft classifier as depending on color to classify cars is never an accurate method because of large divergance in the color of the cars and the large overlap between them and other objects in color tone and nature.

Using this with other features can help to nominate the actual candidates from false positives, however you need to be careful to ensure that the number of feature points from color histogram is much less or at leas comparable to other features to prevent it from having a large vote in the classification process.

Going into color histogram you find that tere exists multiple tuning options such as:
  * Color Space
  * Number of channels used in the selected color space and which channel to use
  * How many Bins you actually use.

I have tried to analyze this visually I have settled that using 32-64 bins provide quite a good feature count and seems to provide the maximum difference in feature pionts between cars and non car images as can be seen in "FeatureInvestigationSpace\ColorSpaceInvestigation" folder.

Now to chose a color space and a channel
The below image show an example of the Car images in the udacity data set, the image is 64 by 64 pixels
![CarHistImg][CarHistImg]
This is the historgram of the above image
![CarHistEx][CarHistEx]
The below image show an example of the Non Car images in the udacity data set, the image is 64 by 64 pixels
![NonCarHistImg][NonCarHistImg]
This is the historgram of the above image
![NonCarHistEx][NonCarHistEx]

Going into this step I was expecting the LAB color space to domincate I used it in previous project and it proved to give very promising results however I can tell that all color space seems to provide a good difference between them but across different examples I can say that the HLS color space and the U and V channels of the LUV color space provide the most variations in the feature space points.

I theorize that the HLS is good due to the fact that car color are usually very sturated although this is not abolutly true as the color saturation usually decreases with time so a 2008 car color will be a lot less saturated than a 2018 car.

For the LUV i think it is due to that it was designed to better dispaly color differeneces and the fact that the illumincae is in a seperate L channel so it would be a lot more robust to different lighting conditions.

The Color Histogram tuneing parameters will be explored with more depth in the Classifier training but for now I settle that I can use the HLS or LUV color space with 32-64 bins and depending on the number of features points of other calssification features I can decide the best number of channels of used.

### 2) Spatial binning:
This is also a soft classifier as using the raw pixel values can give you a rough estimation about the image type but it can never be accurate as there are a lot of factors that can affect pixel values as lighting conditions, camera sensor sensitivity, etc..


Spatial Binning have multiple tuning options such as:
  * Color Space
  * Number of channels used in the selected color space and which channel to use
  * Binning Size
  
The Binning size first, during testing I found that this option actually can result in huge difference in the feature count for the SB feature for example keeping it to 64 result in a whopping 12288 feature points, however downsampling to 8 gives only 192 feature count and the difference in the binning curves is not really this huge as I have observed through a large number of examples

for the color space I observed that again the U and V channels provides a very clear differnece between the car and non-car classes.
This can also be seen in below examples.

The below image show an example of the Car images in the udacity data set, the image is 64 by 64 pixels
![CarSBImg][CarSBImg]
This is the historgram of the above image
![CarSBEx][CarSBEx]
The below image show an example of the Non Car images in the udacity data set, the image is 64 by 64 pixels
![NonCarSBImg][NonCarSBImg]
This is the historgram of the above image
![NonCarSBEx][NonCarSBEx]

### 3) Histogram of Oriented gradients:
This is by far a much better classifier that the above two classifiers as it actually capture the overall strucutre inside the image itself so it can be used with much more confidence that it can better classify the different classes.

For HOG there exists a large number of tuning parameters listed below:
  * Number of orientations: The HOG paper suggest no more than 9 orientations
  * Pixels per cell
  * Cells per block
  * Block Normalization
  * Transformation SQRT
  * Color Space
  * Number of channels used in the selected color space and which channel to use

I gave this option the most time to analyse and invetigate the effect of different parameters, I observed that the parameter values provided by the original picture can give quite a good performance but at least in my experience more optimum values can be foun especially in the orientation count, maybe this is due to the different application as the papaer had the goal to classify pedestarins and the overall strucutre of humans is much different of cars as they can have a lot of curvature and much more sharp edges specially in the bumper and plate areas.

You can view the analysis with much more clarity in the DataVisualization.ipynb last section realted to HOG data visualization.

---

## Second Step: Feature Extraction and Classifier training.

This was done on multiple steps where I tried to manually change the tuning options and train a classifier each time to observe how the accuracy could be affected, however going through with it proved to be much more diffcult than expected and that
