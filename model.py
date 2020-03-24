from __future__ import print_function , division
from builtins import range , input

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

import numpy as np


def resnet_model(class_number):
    resnet = ResNet50V2(include_top=True, weights='imagenet', classes=class_number)
    
    return resnet

def activation_model(resnet):
    activation_layer = resnet.get_layer('post_relu')
    model = Model(inputs = resnet.input , outputs = activation_layer.output)
    return model

