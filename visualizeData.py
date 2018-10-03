import numpy as np
from collections import Counter
from random import shuffle
import cv2
import pandas as pd

def visualize_raw_data(data):
	df = pd.DataFrame(data)
	#print(df.head())
	print("Counts : ")
	print(Counter(df[1].apply(str)))
	print(Counter(df[2].apply(str)))

	for d in data:
		img = d[0]
		output = d[1]
		output2 = d[2]

		cv2.imshow('test', img)
		print(output, output2)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			v2.destroyAllWindows()
			break

def visualize_batch_data(data_x, data_y):
	for i in range(len(data_x)):
		img = data_x[i]
		output = data_y[i]
		cv2.imshow('test', img)
		print(output)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			v2.destroyAllWindows()
			break

if __name__ == "__main__":
	data = np.load('/Users/Rajatagarwal/Desktop/NYU_Academics/Sem_3/CV/Project/input/train_data_2.npy')
	visualize_raw_data(data)

