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

with open("config.json",'r') as file:
    config = json.load(file)

max_last_hundred = 100
last_hundred = []
zero = [0,0,0,0,0]
for i in range(max_last_hundred):
    last_hundred.append(random.choice([keysW,keysS,keysA,keysD,keysR]))


def get_input_from_screen():
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)), dtype=np.float32)[:,:,0:3]
    screen = cv2.resize(screen, (299, 299))
    road_map = extractMap(screen).astype(dtype=np.float32).reshape(1,1,64,64)
    speed = extractSpeed(screen).astype(dtype=np.float32).reshape(1,1,64,64)
    image_fov = restrictFOV(screen).reshape(1,3,299,299)

    return torch.from_numpy(image_fov), torch.from_numpy(road_map), torch.from_numpy(speed)

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
        PressKey(W) if random.uniform(0,1) > 0.5 else None
        return "A"
    if inp == 2:
        PressKey(D)
        PressKey(W) if random.uniform(0,1) > 0.5 else None
        return "D"
    if inp == 3:
        PressKey(S) if random.uniform(0,1) > 0.5 else None
        return "S"
    if inp == 4:
        return "NK"


def play_GTA(model):
    quit = False
    paused = True

    while not quit:
        keys = key_check()
        #keys = 'W'A
        if 'Q' in keys:
            quit = True

        if 'P' in keys:
            paused = not paused
            releaseAll()
            print("PAUSED:",paused)
            time.sleep(1)

        if paused:
            continue
        x1, x2, x3 = get_input_from_screen()
        x4 = torch.from_numpy(np.array(last_hundred).astype(dtype=np.float32).reshape((1,100,5)))
        prediction = model([x1, x2, x3, x4])
        key_pressed = perform_action(prediction.max(1)[1].numpy()[0])
        last_action = [0,0,0,0,0]
        last_action[prediction.max(1)[1].numpy()[0]] = 1

        last_hundred.pop(0)
        last_hundred.append(last_action)
        
        print(prediction.detach().numpy()[0], "Action = ",prediction, key_pressed)


if __name__ == "__main__":

    print("Creating Model Graph")
    model = MainModel(config['model_dict'],5)
    print("Model Created successfully")
    
    if len(sys.argv) < 2:
        print("Please provide the path to load model !!")
        sys.exit()
    print("Loading Model Weights")
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    print("Loaded Model Weights, Starting now !!!")
    play_GTA(model)

