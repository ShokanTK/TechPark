import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def filterKernel(src, kernel):
	"""This function applys kernel to source image"""

	idx_sl = kernel.shape[0] / 2

	dst_cp = np.zeros((src.shape[0] + idx_sl * 2, src.shape[1] + idx_sl * 2))
	dst = np.zeros(dst_cp.shape)

	slicedRow = dst.shape[0] - idx_sl
	slicedCol = dst.shape[1] - idx_sl

	dst_cp[idx_sl:slicedRow, idx_sl:slicedCol] = src

	rows = np.arange(idx_sl, src.shape[0])
	cols = np.arange(idx_sl, src.shape[1])

	for i in rows:
		for j in cols:
			dst[i,j] = (dst_cp[i-idx_sl:i+idx_sl+1, j-idx_sl:j+idx_sl+1]*kernel).sum(axis=1).sum()
	
	dst = dst[idx_sl:slicedRow,idx_sl:slicedCol]

	return dst


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', type=str, help='input image path')


	args = parser.parse_args()

	image_name = args.image

	img = cv2.imread(image_name, 0)
	
	Kernel_K = np.array([[2, 4, 5, 4, 2],
		                   [4, 9, 12, 9, 4],
		                   [5, 12, 15, 12, 5],
		                   [4, 9, 12, 9, 4],
		                   [2, 4, 5, 4, 2]], np.float32) / 159

	Kernel_Gx = np.array([[-1, 0, 1],
		                    [-2, 0, 2],
		                    [-1, 0, 1]], np.float32)

	Kernel_Gy = np.array([[-1, -2, -1],
		                    [0, 0, 0],
		                    [1, 2, 1]], np.float32)

	img_smooth = filterKernel(img, Kernel_K)

	img_smooth = img_smooth.astype(np.uint8)

	img_x = filterKernel(img_smooth, Kernel_Gx)
	img_y = filterKernel(img_smooth, Kernel_Gy)
	img_cover = np.sqrt(np.square(img_x)+np.square(img_y))
	img_cover = img_cover.astype(np.uint8)

	for i in np.arange(img_cover.shape[0]):
		for j in np.arange(img_cover.shape[1]):
			if img_cover[i,j] > 100 and img_cover[i,j] < 200:
				img_cover[i,j] = 255
			else:
				img_cover[i,j] = 0
			# curr = img_cover[i-1:i+2,j-1:j+2]
			# curr[curr < 100] = 0
			# curr[curr > 200] = 0
			# img_cover[i-1:i+2,j-1:j+2]=curr

	# cv2.imwrite("gr_img.jpg", img_cover)

	# cv2.imshow('Original', img)
	# cv2.imshow("Smooth", img_smooth)
	# cv2.imshow('Gradient', img_cover)	

	plt.figure()
	
	plt.subplot(131)
	plt.imshow(img, cmap='gray')
	
	plt.subplot(132)
	plt.imshow(img_smooth, cmap='gray')
	
	plt.subplot(133)
	plt.imshow(img_cover, cmap='gray')
	
	plt.subplot_tool()
	plt.show()
	
	

if __name__ == "__main__":
    main()



