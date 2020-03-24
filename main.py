import tensorflow

from builtins import range , input

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 , preprocess_input , decode_predictions
from tensorflow.keras.preprocessing import image
from model import resnet_model , activation_model
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import glob


if __name__ == '__main__':


# image size is default of resnet
    image_size = (224,244,3)
# image directory
    imgfiles = glob.glob('./*.png')

# output probs (Dense) ,(None, 1000)
    resnet = resnet_model(1000)
# this model is resnet till last conv, not Denses
# output post_relu (Activation) (None, 7, 7, 2048)
    activation_layer = resnet.get_layer('post_relu')
    model = activation_model(resnet)

# getting last layer weights
    final = resnet.get_layer('predictions')
    w = final.get_weights()

# input processing
    x = np.expand_dims(image.img_to_array(image.load_img(imgfiles[2],target_size=(224, 224, 3))),axis = 0)
    x = preprocess_input(x)

# activation outputs
    maps = model.predict(x)
# resner output
    probs = resnet.predict(x)


    detected = decode_predictions(probs)[0]

    pred = int(np.argmax(probs[0]))
    W = w[0][pred]
    cam = maps.dot(w[0].T[pred])
    cam = sp.ndimage.zoom(cam[0], (32,32), order=1)

    plt.subplot(1,2,1);
    plt.imshow(cam , alpha=0.8);
    plt.imshow(x[0] , alpha=0.2);
    plt.subplot(1,2,2);
    plt.imshow(x[0]);
    print(detected[0][1])






