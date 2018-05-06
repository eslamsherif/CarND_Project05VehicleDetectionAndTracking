# CarND_Project05VehicleDetectionAndTracking
Solution for Udacity Self driving car nano degree fifth project: Vehicle Detection and Tracking

---

**Detection of Cars in a Video Stream**

[//]: # (Image References)

[LCS]: https://i.pinimg.com/originals/63/c8/9a/63c89aba0ed994edcfce462b2a4b2b6b.jpg
[CF1]: ./Doc_Images/ColorFiltering/solidWhiteCurve.jpg

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
  
  * Visualize examples of the Dataset:
    Helps to get some sense about the feature spaces and which features can be used to classify the cars correctly.
  * Feature Extraction and Classifier training:
    Try to extract different features and feed them to a classifier to try and 
