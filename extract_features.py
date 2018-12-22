import cv2
import numpy as np
from game_controls import *

def make_purple_white(image,mapping):
    image = image.astype(dtype=np.uint16)
    return_img = np.zeros_like(image[:,:,mapping['G_ch']])
    return_mask = np.logical_and(image[:,:,mapping['G_ch']] + 40 < image[:,:,mapping['R_ch']], image[:,:,mapping['R_ch']] +40 < image[:,:,mapping['B_ch']])
    return_img[return_mask] = 255
    return return_img.astype(dtype=np.uint8)
    
def crop_image(image,h1,h2,w1,w2,hsize,wsize):
    height = image.shape[0]
    width = image.shape[1]

    return cv2.resize(image[h1:h2,w1:w2,:],(hsize,wsize))
    
'''
    Default mapping - BGR
'''
def extractMap(image,ch_mapping = {'R_ch':2,'G_ch':1,'B_ch':0},h1= 248,h2 = 285,w1 = 0, w2 = 60):
    image = crop_image(image,h1,h2,w1,w2,64,64)
    return make_purple_white(image,ch_mapping)
 
def make_blue_white(image,mapping):
    return_img = np.zeros_like(image[:,:,mapping['G_ch']])
    image = image.astype(dtype=np.uint16)

    return_mask = np.logical_and(image[:,:,mapping['B_ch']] > 200,
                            image[:,:,mapping['R_ch']] +50 < image[:,:,mapping['B_ch']],
                            image[:,:,mapping['G_ch']] +50  < image[:,:,mapping['B_ch']])

    return_img[return_mask] = 255
    return return_img.astype(dtype=np.uint8) 
    
def extractSpeed(image,ch_mapping = {'R_ch':2,'G_ch':1,'B_ch':0},h1= 250,h2 = 280,w1 = 220, w2 = 255):
    image = crop_image(image,h1,h2,w1,w2,64,64)
    return make_blue_white(image,ch_mapping)
    
 
def field_of_view(image):
    height = image.shape[0]
    width = image.shape[1]
    fields = np.array([
        [(0,75), (0, 235), (width, 235), (width, 75)],
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, fields, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)

    masked_image = crop_image(masked_image, 75, 235, 0, width, 224, 224)

    return masked_image
 
def restrictFOV(image):
    im = field_of_view(image)
    return im
    
def oneHotLastHundred(last100):
    ret = []
    for x in last100:
        if "R" in x and len(x) > 1:
            x = x.replace("R","")
        ret.append(string2KeyMap.get(x,keysNK))
    return np.array(ret)
    
def getFeatures(data):
    ret = []
    for d in data:
        ret.append([restrictFOV(d[0]),d[1],extractMap(d[0]),extractSpeed(d[0]),oneHotLastHundred(d[3])])
    return ret