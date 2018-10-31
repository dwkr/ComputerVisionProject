import numpy as np
from PIL import ImageGrab
import pyscreenshot as pyscr 
import cv2
import time
import win32api as wapi
import time
import os
from get_keys import key_check
from game_controls import *

STARTING_IDX = 1
DATA_PATH = "D:\\data\\"

max_last_hundred = 100
last_hundred = []
for i in range(max_last_hundred):
    last_hundred.append('')

while True:
    file_name = 'train_data_{}.npy'.format(STARTING_IDX)
    file_name = DATA_PATH + file_name

    if os.path.isfile(file_name):
        print(os.path.abspath(file_name))
        print("File exsits for idx ",STARTING_IDX)
        STARTING_IDX += 1
    else:
        print("File not found ",file_name)
        break

def keys_to_output(keys):
   '''
   Convert keys to a ...multi-hot... array
    0  1  2  3  4   5   6   7    8
   [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
   '''
   output = [0,0,0,0,0]
   output2 = [0,0,0,0,0]

   if 'W' in keys and 'A' in keys:
       output = keysWA
   elif 'W' in keys and 'D' in keys:
       output = keysWD
   elif 'S' in keys and 'A' in keys:
       output = keysSA
   elif 'S' in keys and 'D' in keys:
       output = keysSD
   elif 'W' in keys:
       output = keysW
   elif 'S' in keys:
       output = keysS
   elif 'A' in keys:
       output = keysA
   elif 'D' in keys:
       output = keysD
   elif 'R' in keys:
       output = keysR
   else:
       output = keysNK
    
   if 'W' in keys:
       output2[0] = 1
   if 'A' in keys:
       output2[1] = 1
   if 'S' in keys:
       output2[2] = 1
   if 'D' in keys:
       output2[3] = 1
   if 'R' in keys:
       output2[4] = 1
       
   to_add = 1 * '' + output2[0] * 'W' + output2[1] * 'A' + output2[2] * 'S' + output2[3] * 'D' + output[4] * 'R'
        
   return output, output2, to_add
        
def generateData(file_name, idx):
    train_data = []
    print("Starting in 5 secs")
    time.sleep(5)
    print("Starting!!!!!!")
    last_time = time.time()
    paused = True
    while True:
        keys = key_check()
        if 'P' in keys:
            paused = not paused
            print("Paused: ", paused)
            time.sleep(1)
        
        if not paused:
            last_time = time.time()
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))[:,:,0:3]
            screen = cv2.resize(screen, (299, 299))
            screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
            
            keys_pressed = key_check()
            #if 'A' in keys_pressed or 'D' in keys_pressed or 'S' in keys_pressed:
            output, output2, to_add = keys_to_output(keys_pressed)
            if len(last_hundred) == max_last_hundred:
                last_hundred.pop(0)
            last_hundred.append(to_add)
            train_data.append([screen,output,output2,last_hundred.copy()])
                
            if len(train_data) % 10 == 0:
                print("length ",len(train_data),end='\r')
            if len(train_data) == 500:
                print("Saving")
                np.save(file_name,train_data)
                print("Saved to file - ",file_name)
                idx +=1
                train_data = []
                file_name = 'train_data_{}.npy'.format(idx)
                file_name = DATA_PATH + file_name
                while os.path.isfile(file_name):
                    print(os.path.abspath(file_name))
                    print("File exsits for idx ",STARTING_IDX)
                    idx +=1
                    file_name = 'train_data_{}.npy'.format(idx)
                    file_name = DATA_PATH + file_name

generateData(file_name,STARTING_IDX)