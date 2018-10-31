import numpy as np
import cv2
from PIL import ImageGrab
import pyscreenshot as pyscr
from game_controls import *
from models import *
from directkeys import PressKey, ReleaseKey, W, A, S, D
from get_keys import key_check
import sys
import time
import random

def get_input_from_screen():
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)), dtype=np.float32)[:,:,0:3]
    screen = cv2.resize(screen, (299, 299))
    screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(screen)

def releaseAll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)    
    
# def perform_action(inp):
    # ReleaseKey(W)
    # ReleaseKey(A)
    # ReleaseKey(S)
    # ReleaseKey(D)

    # if inp == 0:
        # PressKey(W)
        # return "W"
    # if inp == 1:
        # PressKey(A)
        # return "A"
    # if inp == 2:
        # PressKey(S)
        # return "S"
    # if inp == 3:
        # PressKey(D)
        # return "D"
    # if inp == 4:
        # PressKey(W)
        # PressKey(A)
        # return "W, A"
    # if inp == 5:
        # PressKey(W)
        # PressKey(D)
        # return "W, D"
    # if inp == 6:
        # PressKey(S)
        # PressKey(A)
        # return "S, A"
    # if inp == 7:
        # PressKey(S)
        # PressKey(D)
        # return "S, D"

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
        PressKey(S)
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
        inp = get_input_from_screen()
        prediction = model(inp)
        key_pressed = perform_action(prediction.max(1)[1].numpy()[0])
        #time.sleep(1)
        print(prediction.detach().numpy()[0], "Action = ",prediction, key_pressed)


if __name__ == "__main__":
    if "Alex" in sys.argv[1]:
        model = AlexNet(int(sys.argv[2]))
    else:
        model = SimpleConvNet(int(sys.argv[2]))
    
    if len(sys.argv) < 2:
        print("Please provide the path to load model !!")
        sys.exit()
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    play_GTA(model)