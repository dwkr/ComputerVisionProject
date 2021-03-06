import numpy as np
import cv2
from PIL import ImageGrab
import pyscreenshot as pyscr
from game_controls import *
from main_model import MainModel
from directkeys import PressKey, ReleaseKey, W, A, S, D
from get_keys import key_check
import sys
import time
import random
from extract_features import extractMap, extractSpeed, restrictFOV
import json
import torch
from GTA_data_set import GTADataset
import threading

THREADING = True;
KEYS_TO_PRESS = set()
PAUSED = True
QUIT = False

with open("config.json",'r') as file:
    config = json.load(file)

with open("stats.json", 'r') as file:
    stats = json.load(file)
    
normalizer_set = GTADataset([],0,0,stats)
    
max_last_hundred = 100
last_hundred = []
zero = [0,0,0,0,0]
for i in range(max_last_hundred):
    last_hundred.append(random.choice([keysW,keysS,keysA,keysD,keysR]))


def get_input_from_screen():
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)), dtype=np.float32)[:,:,0:3]
    screen = cv2.resize(screen, (299, 299))
    screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
    
    road_map = extractMap(screen).astype(dtype=np.float32).reshape(1,1,64,64)
    speed = extractSpeed(screen).astype(dtype=np.float32).reshape(1,1,64,64)
    image_fov = restrictFOV(screen).reshape(1,3,224,224)

    return  torch.from_numpy(normalizer_set.normalize(image_fov,'X1')), torch.from_numpy(normalizer_set.normalize(road_map,'X2')), torch.from_numpy(normalizer_set.normalize(speed,'X3'))

def releaseAll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def perform_action(inp):
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

    if inp == 0:
        PressKey(W)
        return "W"
    if inp == 1:
        PressKey(A)
        #PressKey(W) if random.uniform(0,1) > 0.5 else None
        return "A"
    if inp == 2:
        PressKey(D)
        #PressKey(W) if random.uniform(0,1) > 0.5 else None
        return "D"
    if inp == 3:
        PressKey(S) if random.uniform(0,1) > 0.5 else None
        return "S"
    if inp == 4:
        return "NK"
        
def perform_action_with_threading(inp):
    KEYS_TO_PRESS.clear()

    if inp == 0:
        #PressKey(W)
        #KEYS_TO_PRESS.discard(S)
        KEYS_TO_PRESS.add(W)
        return "W"
    if inp == 1:
        #PressKey(A)
        #KEYS_TO_PRESS.discard(D)
        KEYS_TO_PRESS.add(A)
        KEYS_TO_PRESS.add(W) if random.uniform(0,1) > 0.5 else None
        return "A"
    if inp == 2:
        #PressKey(D)
        #KEYS_TO_PRESS.discard(A)
        KEYS_TO_PRESS.add(D)
        KEYS_TO_PRESS.add(W) if random.uniform(0,1) > 0.5 else None
        return "D"
    if inp == 3:
        #PressKey(S)
        #KEYS_TO_PRESS.discard(W)
        KEYS_TO_PRESS.add(S)
        #if random.uniform(0,1) > 0.5 else None
        return "S"
    if inp == 4:
        #KEYS_TO_PRESS.clear()
        return "NK"


def predictor(model):
    global QUIT
    global PAUSED
    
    while not QUIT:
        keys = key_check()
        #keys = 'W'A
        if 'Q' in keys:
            QUIT = True

        if 'P' in keys:
            PAUSED = not PAUSED
            releaseAll()
            print("PAUSED:",PAUSED)
            time.sleep(1)

        if PAUSED:
            continue
        x1, x2, x3 = get_input_from_screen()
        x4 = torch.from_numpy(np.array(last_hundred).astype(dtype=np.float32).reshape((1,100,5)))
        #print(x1.shape,type(x1))
        #cv2.imshow('image',x1.detach().numpy().reshape(299,299,3))
        #cv2.imshow('map',x2.detach().numpy().reshape(64,64,1))
        #cv2.imshow('speed',x3.detach().numpy().reshape(64,64,1))
        
        #if(cv2.waitKey(25) & 0xFF == ord('q')):
        #    cv2.destroyAllWindows()
        #    break
        #input("Press Enter to continue...")
        
        prediction = model([x1, x2, x3, x4])
        if not THREADING:
            key_pressed = perform_action(prediction.max(1)[1].numpy()[0])
        else:
            key_pressed = perform_action_with_threading(prediction.max(1)[1].numpy()[0])
            
        last_action = [0,0,0,0,0]
        last_action[prediction.max(1)[1].numpy()[0]] = 1

        last_hundred.pop(0)
        last_hundred.append(last_action)
        
        print(prediction.detach().numpy()[0],"\n", "Action = ", key_pressed)

def player():


    while not QUIT:
        
        if PAUSED:
            continue
        print("KEYS_TO_PRESS",KEYS_TO_PRESS)
        for key in KEYS_TO_PRESS:
            PressKey(key)
        time.sleep(0.5)
        releaseAll()
        #time.sleep(0.7) # GOOD FOR DEBUG
        time.sleep(0.6)

            
            
if __name__ == "__main__":

    print("Creating Model Graph")
    model = MainModel(config['model_dict'],5)
    model.eval()
    print("Model Created successfully")
    
    if len(sys.argv) < 2:
        print("Please provide the path to load model !!")
        sys.exit()
    print("Loading Model Weights")
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    print("Loaded Model Weights, Starting now !!!")
    if not THREADING:
        predictor(model)
    else:
        
        pred = threading.Thread(name='predictor', target=predictor, args=(model,))
        pred.start()
        plyr = threading.Thread(name='player', target=player)
        plyr.start()

