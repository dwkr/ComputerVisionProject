import numpy as np
import cv2
import pyscreenshot as pyscr
from game_controls import *
from models import *
from directkeys import PressKey, ReleaseKey, W, A, S, D

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
    if inp == 1:
        PressKey(A)
    if inp == 2:
        PressKey(S)
    if inp == 3:
        PressKey(D)
    if inp == 4:
        PressKey(W)
        PressKey(A)
    if inp == 5:
        PressKey(W)
        PressKey(D)
    if inp == 6:
        PressKey(S)
        PressKey(A)
    if inp == 7:
        PressKey(S)
        PressKey(D)


def play_GTA(model):
    while(True):
        inp = get_input_from_screen()
        prediction = model(inp)
        print(prediction)
        perform_action(prediction.max(1)[1].numpy()[0])

if __name__ == "__main__":
    model = SimpleConvNet(9)
    model.load_state_dict(torch.load('/Users/Rajatagarwal/Desktop/NYU_Academics/Sem_3/CV/Project/models/SimpleConvNet')) 
    play_GTA(model)


