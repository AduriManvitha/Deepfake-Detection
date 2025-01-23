import cv2
import os
import numpy as np
from keras.models import Sequential, load_model
from tensorflow.keras.utils import load_img
from keras import backend as k
from keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import image_to_arry,load_img


def fake_real_detection(img_path):

    k.clear_session()

    model_path = 'mobilenet_model.h5'
    model = load_model(model_path)
    #img_path = img_path
    print("img=",img_path)
    testing_img=cv2.imread(img_path)
    cv2.imwrite("/Users/apple/Desktop/DeepFake/static/detection.jpg", testing_img)

    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result="Real"
    else:
        result="Fake"

    k.clear_session()

    return result

#fake_real_detection("C:/Users/DeepFake/images_folder/real_10002.jpg")