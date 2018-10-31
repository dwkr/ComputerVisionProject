import numpy as np
import cv2

DATA_PATH = "D:\\data\\"

data = np.load(DATA_PATH+"train_data_53.npy")
for d in data:
    cv2.imshow('window', d[0])
    print(d[1])
    print(d[2])
    print(d[3])
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break