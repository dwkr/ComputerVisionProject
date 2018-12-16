import numpy as np
from game_controls import *
import glob

def load_data(input_path, balance, shuffe_data = True):
    print("Reading files from ", input_path)
    files = glob.glob(input_path)
    print("Number of files available : ", len(files))
    all_data = []
    stats = {'straight':0,'left':0,'right':0,'reduce':0,'Wait(R)':0}
    for file in files:
        print("Reading", file)
        data = np.load(file)
        for d in data:
            
            if d[1] == keysW:
                stats['straight'] +=1
            elif d[1] == keysWA or d[1] == keysA:
                stats['left'] +=1
            elif d[1] == keysWD or d[1] == keysD:
                stats['right'] +=1
            elif d[1] == keysR:
                stats['Wait(R)'] +=1
            else:
                stats['reduce'] +=1
                    

    return stats

if __name__ == "__main__":
    print(load_data("/data/cvision_fa18/ra2630/Project/data/train_data_*.npy",balance=True))
