import numpy as np
from PIL import ImageGrab
import pyscreenshot as pyscr 
import cv2
import time

last_time = time.time()
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))[:,:,0:3] # TODO: change as function of scrreen res
    print('Loop took {} seconds'.format(time.time() - last_time))
    last_time = time.time()
    cv2.imshow('window', cv2.cvtColor(cv2.resize(screen, (240, 240)), cv2.COLOR_BGR2RGB))
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break