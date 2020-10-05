# BrainTumourDetection
Image classification based on brain tumour scans

This is a project about detecting if there is a tumor in the brain given the MRI scan. 

I choose these 2 attached datasets because there are a lot of data in them and have some sort of the same format and dimension in them.

I can take the images and reformat the pixels to grayscale to reduce the size of each image

I have also filter the images of the dataset to limit myself to the top view of the brain only.

Question 1: the size of brain in each image is different?
Question 2: some tumours appear as white, some tumours appear as black, kinda hard to identify tumours in various dataset?

I look at some algorithms about image processing for grayscale or intensity, many suggests CNN (Convolutional Neural Network)

I am thinking about KNN algorithm to group the images with the same reponse since my dataset is kinda large I would say, I can increase my value of k to have a better prediction, but not sure how to convert my images to datapoints that could calculate the distance between them.

My idea is have a website that take the brain's top view image and output to the user about my prediction.

If the user doesn't have the tumor, we congrate them; otherwise, we can suggest some local hospitals near them.

The purpose is to give the user a second opinion about their brain scan since doctors can mistakes about their analysis on the image.
