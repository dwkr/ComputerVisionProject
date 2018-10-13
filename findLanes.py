import cv2
import numpy as np
import matplotlib.pyplot as plt

def field_of_view(image):
	height = image.shape[0] 
	width = image.shape[1]
	fields = np.array([
		[(0,height), (92, 133), (216, 133), (width, height)],
		[(50,height), (75, 150), (210,150), (220, height)]
		])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, fields, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def display_lines(img, lines):
	line_image = np.zeros_like(img)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
				cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)

	return cv2.addWeighted(img, 0.8, line_image, 1, 1)

def generate_coordinates(image, line_parameters):
	slope, y_intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - y_intercept)/slope)
	x2 = int((y2 - y_intercept)/slope)
	return np.array([x1,y1,x2,y2])

def avg_slope_intercept(img, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1,x2,y1,y2 = line.reshape(4)
		params = np.polyfit((x1,x2), (y1,y2), 1) #Find Slope and Y-intercept of line
		slope = params[0]
		y_intercept = params[1]

		if slope < 0:
			left_fit.append((slope, y_intercept))
		elif slope > 0:
			right_fit.append((slope, y_intercept))
	if len(left_fit) > 0:
		left_fit_avg = np.average(left_fit, axis = 0)
	else:
		left_fit_avg = None

	if len(right_fit) > 0:
		right_fit_avg = np.average(right_fit, axis = 0)
	else:
		right_fit_avg = None

	if left_fit_avg is None:
		left_lane = np.array([0,0,0,0])
	else:
		left_lane = generate_coordinates(img, left_fit_avg)
	if right_fit_avg is None:
		right_lane = np.array([0,0,0,0])
	else:
		right_lane = generate_coordinates(img, right_fit_avg)

	return (left_lane, right_lane)
	

def find_lanes(img):
	lane_image = np.copy(img)
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	masked_image = field_of_view(canny)
	return masked_image
	lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
	lanes = avg_slope_intercept(img, lines)
	line_image = display_lines(img, lanes)
	return line_image

def visualize_raw_data(data):
	for d in data:
		img = d[0]
		lane_img = find_lanes(img)
		output = d[1]
		output2 = d[2]
		cv2.imshow('test', lane_img)
		print(output, output2)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			v2.destroyAllWindows()
			break

if __name__ == "__main__":
	data = np.load('/Users/Rajatagarwal/Desktop/NYU_Academics/Sem_3/CV/Project/input/train_data_5.npy')
	visualize_raw_data(data)

