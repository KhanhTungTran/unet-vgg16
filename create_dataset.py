# import the necessary packages
from imutils import paths, resize
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import argparse
import cv2
import os
from random import seed
from random import randint, uniform
from math import ceil, floor
from tqdm import tqdm

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
    help="path to the watermark directory of images (assusmed to be transparent PNG)")
ap.add_argument('-i', '--input', required=True,
    help="path to the input directory of images")
ap.add_argument('-oo', '--output_original', required=True,
    help="path to the original image output directory")
ap.add_argument('-oi', '--output_image', required=True,
    help="path to the image output directory")
ap.add_argument("-n", "--number", type=int, default=5000,
	help="number of images to generate")	
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
ap.add_argument("-s", "--seed", type=int, default=1,
	help="seed used to randomize the images")
args = vars(ap.parse_args())

seed(args["seed"])

count = 1
list_watermarks = list(paths.list_images(args["watermark"]))
list_watermarks.sort()
for watermark_path in list_watermarks:
	print(watermark_path)
	# load the watermark image, making sure we retain the 4th channel
	# which contains the alpha transparency
	watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
	(wH, wW) = watermark.shape[:2]
	print(watermark.shape)
	# cv2.imshow("watermark", watermark)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	if args["correct"] > 0:
		(B, G, R, A) = cv2.split(watermark)
		B = cv2.bitwise_and(B, B, mask=A)
		G = cv2.bitwise_and(G, G, mask=A)
		R = cv2.bitwise_and(R, R, mask=A)
		watermark = cv2.merge([B, G, R, A])

	# not_trans_mask = watermark[:, :, 3] != 0
	# watermark[not_trans_mask] = [not_trans_mask[3], 0, 0, 0]

	# not_zero_mask = watermark[:, :, 0] != 0
	# not_255_mask =  watermark[:, :, 0] != 255
	# watermark[not_zero_mask] = [100, 100, 100, 26]
	# watermark[not_255_mask] = [200, 200, 200, 26]
	# watermark[:, :, 3] = 255

	# cv2.imshow("watermark", watermark)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	orig_watermark = watermark
	# cv2.imshow("watermark", watermark)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# NOTE: random input images
	list_images_path = [format(randint(1, 17125), '07d') + '.jpg' for _ in range(args["number"])]
	# loop over the input images
	for image_path in tqdm(list_images_path):
		# print(image_path)
		# load the input image, then add an extra dimension to the
		# image (i.e., the alpha transparency)
		image = cv2.imread(args["input"] + '/' + image_path)
		(h, w) = image.shape[:2]
		image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
		# construct an overlay that is the same size as the input
		# image, (using an extra dimension for the alpha transparency),
		# then add the watermark to the overlay in the bottom-right
		# corner
		# overlay = image.copy()
		overlay = np.zeros((h, w, 4), dtype="uint8")

		# NOTE: random size of watermark and random location
		new_width = randint(min(int(w*3/5), 150), max(int(w*3/5), 150))
		watermark = resize(orig_watermark, width=new_width)
		(wH, wW) = watermark.shape[:2]
		while int(wH/2) >= h-int(wH/2)-1 or int(wW/2) >= w-int(wW/2)-1:
			new_width = randint(30, new_width)
			watermark = resize(orig_watermark, width=new_width)
			(wH, wW) = watermark.shape[:2]
		y_center = randint(int(wH/2), h-int(wH/2)-1)
		x_center = randint(int(wW/2), w-int(wW/2)-1)

		# not_zero_mask = watermark[:, :, 0] != 0
		# not_255_mask =  watermark[:, :, 0] != 255
		# watermark[not_zero_mask] = [0, 0, 0, 26]
		# watermark[not_255_mask] = [75, 75, 75, 26]
		overlay[y_center - floor(wH/2):y_center + ceil(wH/2), x_center - floor(wW/2):x_center + ceil(wW/2)] = watermark
		# blend the two images together using transparent overlays
		output = image.copy()

		# NOTE: Random alpha
		alpha = uniform(0.5, 0.75)
		cv2.addWeighted(overlay, alpha, output, 1, 0, output)

		# NOTE: write the output image to disk
		x = [0]*4
		x[1] = y_center - floor(wH/2)
		x[3] = y_center + ceil(wH/2)
		x[0] = x_center - floor(wW/2)
		x[2] = x_center + ceil(wW/2)
		c1 = max(int(min(x[0], x[2])-abs(x[2]-x[0])*0.25), 0), max(int(min(x[1], x[3])-abs(x[3]-x[1])*0.25), 0)
		c2 = min(int(max(x[0], x[2])+abs(x[2]-x[0])*0.25), image.shape[1]), min(int(max(x[1], x[3])+abs(x[3]-x[1])*0.25), image.shape[0])

		image_file_name = format(count, '07d') + ".png"
		label_file_name = format(count, '07d') + ".txt"
		cv2.imwrite(os.path.sep.join((args["output_original"], image_file_name)), image[c1[1]:c2[1], c1[0]:c2[0]])
		p = os.path.sep.join((args["output_image"], image_file_name))
		cv2.imwrite(p, output[c1[1]:c2[1], c1[0]:c2[0]])

		count += 1
