import numpy as np
from PIL import ImageGrab
import pyscreenshot as pyscr 
import cv2
import time
import win32api as wapi
import time
import os
from get_keys import key_check

keysWA = [0,0,0,0,1,0,0,0,0]
keysWD = [0,0,0,0,0,1,0,0,0]
keysSA = [0,0,0,0,0,0,1,0,0]
keysSD = [0,0,0,0,0,0,0,1,0]
keysW = [1,0,0,0,0,0,0,0,0]
keysA = [0,1,0,0,0,0,0,0,0]
keysS = [0,0,1,0,0,0,0,0,0]
keysD = [0,0,0,1,0,0,0,0,0]
keysNK = [0,0,0,0,0,0,0,0,1]

STARTING_IDX = 1

while True:
	file_name = 'train_data_{}.npy'.format(STARTING_IDX)
	
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
   output = [0,0,0,0,0,0,0,0,0]
   output2 = [0,0,0,0]

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
		
   return output, output2
		
def generateData(file_name, idx):
	train_data = []
	print("Starting in 5 secs")
	time.sleep(5)
	print("Starting!!!!!!")
	last_time = time.time()
	paused = False
	while True:
		keys = key_check()
		if 'P' in keys:
			paused = not paused
			print("Paused: ", paused)
		
		if not paused:
			last_time = time.time()
			screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))[:,:,0:3]
			screen = cv2.resize(screen, (299, 299))
			screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
			
			keys_pressed = key_check()
			output, output2 = keys_to_output(keys_pressed)
			
			train_data.append([screen,output,output2])
			if len(train_data) % 10 == 0:
				print("length ",len(train_data))
			if len(train_data) == 500:
				print("Saving")
				np.save(file_name,train_data)
				print("Saved to file - ",file_name)
				idx +=1
				train_data = []
				file_name = 'train_data_{}.npy'.format(idx)
				
		
	
		

generateData(file_name,STARTING_IDX)