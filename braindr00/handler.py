import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import simplejson as json

import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')


import tensorflow
import keras

from PIL import Image
from io import BytesIO
import base64
import numpy as np
from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import urllib


def reshape(data, n=256):
    return np.pad(data, (((n-data.shape[0])//2, ((n-data.shape[0]) + (data.shape[0]%2 >0))//2), 
                             ((n-data.shape[1])//2, ((n-data.shape[1]) + (data.shape[1]%2 >0))//2)), 
                      "constant", constant_values = (0,0))


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def load_model(**kwargs):
    model = keras.models.load_model("/root/function/braindr_model00.h5")
    return model


def download_image(inp, out):
    urllib.urlretrieve(inp, out)
    return out

def load_base64(b64):
    im = Image.open(BytesIO(base64.b64decode(b64)))
    im_arr = np.asarray(im)
    im_arr = rgb2gray(im_arr)
    im_arr = reshape(im_arr, np.max(im_arr.shape))
    im_arr = resize(im_arr, (256,256))
    return im_arr[:,:,None]

def load_image(input_file):
    im_arr = plt.imread(input_file)
    im_arr = rgb2gray(im_arr)
    im_arr = reshape(im_arr, np.max(im_arr.shape))
    im_arr = resize(im_arr, (256,256))
    return im_arr[:,:,None]


def main(input_images):
    data = np.zeros((len(input_images), 256, 256, 1))
    for i,input_image in enumerate(input_images):
        if input_image.startswith("http"):
            suffix = input_image.split(".")[-1]
            assert suffix in ["JPG", "jpg","jpeg","JPEG","png","PNG"]
            input_file = download_image(input_image, "data.{}".format(suffix))
            data[i,:,:,:] = load_image(input_file)
        else:
            data[i,:,:,:] = load_base64(input_image)
    
    model = load_model()
    return model.predict(data).tolist()


def handle(st):
    inp = json.loads(st)
    print(json.dumps(main(**inp)))
    # print("model loaded")
