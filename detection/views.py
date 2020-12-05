from django.shortcuts import render
import io
from tensorflow.keras import models
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# Create your views here.

def index(request):
    data = dict()
    data["title"] = "AI Brain Tumour Detection"
    data["message"] = "Please submit a top view of your brain MRI scan to detect if you have a tumour"
    return render(request, "welcome.html", data)

def prepare_image(image, target):
    image = image.convert('L')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def evaluate_image(request):
    if request.method == "POST":
        img = request.FILES["mri-scan"].read()
        img = Image.open(io.BytesIO(img))
        img = prepare_image(img, (300, 400))
        model = models.load_model('modelCNN.h5')
        outcome = model.predict(img)
        outcome = outcome[0][0]
        data = dict()
        if outcome > 0.5:
            data["title"] = "POSITIVE"
            data["message"] = """The predicted outcome is positive, we would recommend that you get professional help\n
                                 This is only for reference and don't be too woried!!!!"""
            return render(request, "outcome.html", data)
        else:
            data["title"] = "NEGATIVE"
            data["message"] = """The predicted outcome is negative. Congratulations!!!! We would still recommend to get professional help\n
                                  This is only for reference and stay healthy!!!!"""
            return render(request, "outcome.html", data)