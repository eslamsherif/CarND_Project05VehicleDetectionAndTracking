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

[CarHOGImg]: ./ReportImages/HOG/CarImage.png
[CarHOGEx]: ./ReportImages/HOG/CARRGB_CH0.png
[NonCarHOGImg]: ./ReportImages/HOG/NonCarImage.png
[NonCarHOGEx]: ./ReportImages/HOG/NONCARRGB_CH0.png

[Visualization]: ./ReportImages/SlidingWindow/Visualization.png
[ClassifierOutput]: ./ReportImages/SlidingWindow/ClassifierOutput.png
[HeatMap]: ./ReportImages/SlidingWindow/HeatMap.png

[FalsePositives]: ./ReportImages/AreasOfImprovement/FalsePositives.png

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

Now to chose a color space and a channel
The below image show an example of the Car images in the udacity data set, the image is 64 by 64 pixels
![CarHOGImg][CarHOGImg]
This is the HOG of R channel of RBG color space of the above image
![CarHOGEx][CarHOGEx]
The below image show an example of the Non Car images in the udacity data set, the image is 64 by 64 pixels
![NonCarHOGImg][NonCarHOGImg]
This is the HOG of R channel of RBG color space of the above image
![NonCarHOGEx][NonCarHOGEx]

You can view the analysis with much more clarity in the DataVisualization.ipynb last section realted to HOG data visualization.

I have settled on using the LUV color space as well and all channels are being used.

---

## Second Step: Feature Extraction and Classifier training.

This was done on multiple steps where I tried to manually change the tuning options and train a classifier each time to observe how the accuracy could be affected, however going through with it proved to be much more diffcult than expected and that it would consume long time to achieve.

I developed one function "ParameterTuning" that takes a list of lists, each internal list contains the parameter values for a single trial.

To generate this Tuning Prameters list, I decided to go with the following flow:
  * I write the parameter list I want to search in a xlsx file "Feature_Extraction_Tuning.xlsx"
  * I save the xlsx file as a csv file "Feature_Extraction_Tuning.csv" for easier parsing
  * I developed a python script "Extract_Parameter_Search_Lists.py" that parse the csv file and output the list that I copy and paste to the function list so I can explore the feature space easily.
  
After finishing the implementation I found that the classifier accuracy can vary from run to run which was weird untill I understood that the "TrainLinearSVCModel" function contain a random number generator that is used during the training and you can't control it's seed, as such I added a loop so that each classifier is trained a configurable number of times and I average the accuracies manually later.

Please view the result of the training in the file "Feature_Extraction_Tuning.xlsx" in the output field.

I had to go on several trials, each one is shown in a seperate worksheet and the final tuning parameters were chosen were at index 2 in work sheet "LinearSVC_Comb"

The classifier accuracy obtained was 99.09% and the classifier was dumped to a pkl file "TrainedClassifier0.pkl" along with all the tuning parameters used to train this classifier.

---

## Third Step: Use Classifier to detect objects in test Images and Videos.

In Notebook "TestClassifierAndVideoDetections.ipynb" I have used the dumped classifier in previous method "TrainedClassifier0.pkl".

I collect the Feature Extraction parameter from the pkl file, However We need to devise the method to collect the image patchs to extract the features from.

In the lessons they recommended using the sliding window technique, for this I need to define the different start, end positions and the scale of each widnow.

This was mainly done by testing, setting the windows parameters at very low values will lead to very accurate detections but the run time will be very long because for each window the feature needs to be extracted and then feed to the classifier and so on, this will lead to a very large number of overlapping postivites so it can be assured that the objects are detected accuratly, give that the classifier is accurate enough.

While on the other hand decreasing the window parameters will lead to a smal run time but the detections will be not very robust as you can miss some of the important features.

So it is basically a trade off between accuracy and run time as usual in these typs of applications.

The lessons offered one function "draw_boxes" to use in drawing boxes, basically a wrapper for the cv2 function. I have modified the implementation to be able to see the sliding window range and scale I am using similar to below output
![Visualization][Visualization]

After lots of testing with a lot of window parameters, I settld on having 4 scales (32 * 32), (64 * 64), (128 * 128), (192 * 192)

all of the scales are scaled to 64 * 64 then fed to the featrue extractor and the classifier and the classifier output is recorded.

The output of this stage is similar to the below image
![ClassifierOutput][ClassifierOutput]

After that we need to gorup all the widnow outputs to meanifngul groups each one would represent a car.

I used heat maps as recommended in the Lessons, so The Output of all windows are accumlated and then Thresholded by a configurabe value to ensure the removal of false positives.

The Heat maps produce gorupings similar to the below output image:
![HeatMap][HeatMap]

Here we can see in the third column the final output of the image and how it is much more noise free than the classifier output.

The same method is applied to the test video and project video and the output can be found under videos "test_video_out.mp4" and "project_video_out.mp4"

---

## Areas of Improvement:

### Data Set:

1) The Output video confirms the result of visual analysis of the data set that the less dark cars (white in this case) detection are much less robust than the dark colored ones, This can be solved by complementing the existing data set with more images with less toned colores and more bright ones.

2) The Classifier Although has a very high output (99.1%) still shows a lot of false positivis in very weird areas as can be shown in the below image
![FalsePositives][FalsePositives]

The large number of detections on pavement seems to be a very good indicator for me that the classifier is overfitting, using the lesson recommendation to manually seperate the time series images to ensure a large varity between te training and validation set should solve this issue.

3) The Run time is quite large this can be addressed with the following:
  * Change parameter of the feature extractor to have a smaller number of features, as seen in the "Feature_Extraction_Tuning.xlsx" in the last few combinatinos had very similar results, for example a classifier with 2040 feature count accuracy was 98.8% while a classifier with only 1200 feature count accuracy was 98.6%.
  Decrasing the number of features should allow for smaller processing time for each subsample in the image.
  * Extracing the feature once for the whole image and then subsample the output isntead of extracting for each image subsample one by one.
  This was discussed in the lessons to allow having a smaller run time by removing a lot of redundant computations from extracting the same features again and again.
  * Introducing tracking to the solution
  Up till now we are running the classifier one each frame in the video although it will be usually very similar if not nearly identical to the previous frame, thus redoing all the computation again.
  One way to solve this is to start tracking the vehicles in each frame, so instead of running the classifier on each frame we can run it on one or two frames and then say this output is sufficient and begin tracking the detections inside it.
  You can compute the optical flow between frames to detect whenever a large change have actually occured then you can re run the classifier on the image to re-evaluate your tracked detections.
  
---

## Concolusion:

The Cars are detected in the video correctly for nearly the whole frame, False positives are kept to minimal and we can start to work on the mentioned above areas of improvement to improve the output accuracy and run time beyond the current level.
