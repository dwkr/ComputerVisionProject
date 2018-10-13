import numpy as np
import cv2
import pyscreenshot as pyscr
from game_controls import *
from models import *
from directkeys import PressKey, ReleaseKey, W, A, S, D
from get_keys import key_check
import sys

def get_input_from_screen():
    screen = np.array(pyscr.grab(bbox=(0,40,800,640)), dtype=np.float32)[:,:,0:3]
    screen = cv2.resize(screen, (299, 299))
    screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(screen)

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
        return "A"
    if inp == 2:
        PressKey(S)
        return "S"
    if inp == 3:
        PressKey(D)
        return "D"
    if inp == 4:
        PressKey(W)
        PressKey(A)
        return "W, A"
    if inp == 5:
        PressKey(W)
        PressKey(D)
        return "W, D"
    if inp == 6:
        PressKey(S)
        PressKey(A)
        return "S, A"
    if inp == 7:
        PressKey(S)
        PressKey(D)
        return "S, D"


def play_GTA(model):
    quit = False

    while not quit:
        inp = get_input_from_screen()
        prediction = model(inp)
        key_pressed = perform_action(prediction.max(1)[1].numpy()[0])
        print(prediction.detach().numpy()[0], "Action = ",prediction, key_pressed)
        keys = key_check()
        keys = 'W'
        if 'Q' in keys:
            quit = True

if __name__ == "__main__":
    model = SimpleConvNet(9)
    if len(sys.argv) < 2:
        print("Please provide the path to load model !!")
        sys.exit()
    model.load_state_dict(torch.load(sys.argv[1]))
    play_GTA(model)