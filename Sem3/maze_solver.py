import cv2
import argparse
import numpy as np






def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', type=str, help='input image path')

	args = parser.parse_args()

	image_name = args.image
	img = cv2.imread(image_name)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	path = np.zeros(gray.shape, dtype=np.uint8)
	
	cv2.drawContours(path, contours[1], 0, (255, 255, 255), cv2.FILLED)

	kernel = np.ones((21, 21), dtype=np.uint8)
	
	path = cv2.dilate(path, kernel)

	path_erode = cv2.erode(path, kernel)

	cv2.absdiff(path, path_erode, path)

	channels = cv2.split(img)
	channels[0] &= ~path
	channels[1] &= ~path
	channels[2] |= path

	dst = cv2.merge(channels)
	
	cv2.imshow('img',dst)
	cv2.waitKey(0)

	

if __name__ == '__main__':
	main()