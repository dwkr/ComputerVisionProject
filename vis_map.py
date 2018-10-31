from extract_features import *
import matplotlib.pyplot as plt
import glob
import time


if __name__ == "__main__":

    DATA_PATH = "D:\\data\\"
    data = np.load(DATA_PATH+"train_data_125.npy")

    h1,h2,w1,w2 = 248,285,0,60
    ch_mapping = {'R_ch':2,'G_ch':1,'B_ch':0}

    files = glob.glob(DATA_PATH+'*npy')
    print("Number of files available : ", len(files))
    for file in files:
        data = np.load(file)
        print(file)
        
        for d in data:
            #plt.figure(figsize=(10,10))
            d_rgb = d[0]#cv2.cvtColor(d[0],cv2.COLOR_BGR2RGB)
            #plt.imshow(d[0])
            #mp = crop_image(d_rgb,h1,h2,w1,w2)
            #plt.imshow(mp)
            #plt.plot
            #plt.figure()
            #Y = np.zeros_like(mp[:,:,1])
            #Y = mp[:, :, 0] * 65.481 + mp[:, :, 1] * 128.553 + mp[:, :, 2] * 24.966 + 16
            #plt.imshow(make_purple_white(mp), cmap='gray')
            bin_map = extractMap(d[0],ch_mapping)#make_purple_white(mp,ch_mapping)
            cv2.imshow('map',bin_map)
            cv2.resizeWindow('map',2*bin_map.shape[0],2*bin_map.shape[1])
            bin_speed = extractSpeed(d[0],ch_mapping)#make_purple_white(mp,ch_mapping)
            cv2.imshow('speed',bin_speed)
            cv2.resizeWindow('speed',2*bin_speed.shape[0],2*bin_speed.shape[1])
            FOV = restrictFOV(d[0])#make_purple_white(mp,ch_mapping)
            cv2.imshow('FOV',FOV)
            cv2.resizeWindow('FOV',2*FOV.shape[0],2*FOV.shape[1])
            #plt.plot()
            
            cv2.imshow('iamge',d_rgb)#cv2.cvtColor(d_rgb,cv2.COLOR_RGB2BGR))
            #cv2.resizeWindow('map',2*bin_map.shape[0],2*bin_map.shape[1])
            #plt.plot() 
            
            print(d[1])
            print(d[2])
            print(d[3])
            print(oneHotLastHundred(d[3]))
            
            if(cv2.waitKey(25) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break

