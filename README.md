# BrainTumourDetection
Image classification based on brain tumour scans

This is a project about detecting if there is a tumor in the brain given the MRI scan. 

I choose these 2 attached datasets because there are a lot of data in them and have some sort of the same format and dimension in them.

I can take the images and reformat the pixels to grayscale to reduce the size of each image

I have also filter the images of the dataset to limit myself to the top view of the brain only.

To run the django project: 

1. Make sure to have virtualenv installed
2. Activate virtualenv venv
3. pip install -r requirements.txt
4. python manage.py runserver

The website will be deployed from the above steps and just follow the instructions in the website after.
