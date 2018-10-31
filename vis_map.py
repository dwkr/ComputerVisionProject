import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

DATA_PATH = "D:\\data\\"
data = np.load(DATA_PATH+"train_data_125.npy")

h1,h2,w1,w2 = 248,285,0,60

def make_purple_white(image):
    image = image.astype(dtype=np.uint16)
    return_img = np.zeros_like(image[:,:,1])
    return_mask = np.logical_and(image[:,:,1] + 40 < image[:,:,0], image[:,:,0] +40 < image[:,:,2])
    print(return_mask.shape)
    return_img[return_mask] = 255
    return return_img.astype(dtype=np.uint8)
    
def crop_image(image,h1,h2,w1,w2):
    height = image.shape[0]
    width = image.shape[1]

    return image[h1:h2,w1:w2,:]

files = glob.glob(DATA_PATH+'*npy')
print("Number of files available : ", len(files))
for file in files:
    data = np.load(file)

    for d in data:
        #plt.figure(figsize=(10,10))
        d_rgb = cv2.cvtColor(d[0],cv2.COLOR_BGR2RGB)
        #plt.imshow(d[0])
        mp = crop_image(d_rgb,h1,h2,w1,w2)
        #plt.imshow(mp)
        #plt.plot
        #plt.figure()
        #Y = np.zeros_like(mp[:,:,1])
        #Y = mp[:, :, 0] * 65.481 + mp[:, :, 1] * 128.553 + mp[:, :, 2] * 24.966 + 16
        #plt.imshow(make_purple_white(mp), cmap='gray')
        bin_map = make_purple_white(mp)
        cv2.imshow('map',bin_map)
        cv2.resizeWindow('map',2*bin_map.shape[0],2*bin_map.shape[1])
        #plt.plot()
        
        cv2.imshow('iamge',cv2.cvtColor(d_rgb,cv2.COLOR_RGB2BGR))
        #cv2.resizeWindow('map',2*bin_map.shape[0],2*bin_map.shape[1])
        #plt.plot() 
        
        print(d[1])
        print(d[2])
        print(d[3])
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break