# Report 1

Objective: Detecting brain tumor based on the top view of MRI scan

The dataset remains the same as before, I preprocess my data by resizing the images to the same dimension in order to use CNN.

I use Convolutional Neuron Network to train my data in order to recognize the tumor in the brain and use an architecture in CNN called VGGnet which is for large scale image recognition

I split my dataset into training, validation and testing respectively for 0.64, 0.16, 0.2 which is quite standard.

I also use the loss function binary_crossentropy which is specifically for classification problem and since tumor detection is related to medecine, I use False positive rate to rate my model's performance.

I still am not able to run the code due to conflict of shape in the training data. Originally I got error on dimension since it layer.Conv2D expects a 4d input, but I solve it by adding column of 1 in the beginning.

Since I still have bugs in my code, I couldn't really explain overfitting and underfitting.

The problem obviously in my project is that my code doesn't work but also while I am reducing the size or increasing the size of images, I loss crucial information on the data which can greatly impact my performance.

Maybe I can try with Fully Convolutional Network since it doesn't require the images to be the same dimension.
