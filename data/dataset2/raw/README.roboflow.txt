
crack - v2 2022-09-29 2:14pm
==============================

This dataset was exported via roboflow.com on September 29, 2022 at 6:15 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 4029 images.
Concrete are annotated in COCO Segmentation format.

The following pre-processing was applied to each image:
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 30 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 1 pixels


