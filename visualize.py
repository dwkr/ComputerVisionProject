import numpy as np
import cv2

data = np.load("train_data_3.npy")
for d in data:
    cv2.imshow('window', d[0])
    print(d[1])
    print(d[2])
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break