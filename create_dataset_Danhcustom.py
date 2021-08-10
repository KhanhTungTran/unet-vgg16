# import the necessary packages
from imutils import paths, resize
import sys
import numpy as np
import argparse
import cv2
import os
from random import seed
from random import randint, uniform
from math import ceil, floor
from tqdm import tqdm
import random

np.set_printoptions(threshold=sys.maxsize)


# Hàm này dùng để tạo ảnh chứa thông tin của alpha channel của watermark
def create_alpha_img(watermark_path):
    watermark = cv2.imread(watermark_path, -1)
    alpha = cv2.split(watermark)[3]
    alpha = np.dstack([alpha, alpha, alpha])
    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
                help="path to the watermark directory of images (assumed to be transparent PNG)")
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
    alpha = create_alpha_img(watermark_path)
    watermark = cv2.imread(watermark_path)
    (wH, wW) = watermark.shape[:2]

    # NOTE: random input images
    list_images_path = [str(format(randint(0, 17125), '07d')) + '.jpg' for _ in range(args["number"])]
    # loop over the input images
    for image_path in tqdm(list_images_path):
        try:
            image = cv2.imread(args["input"] + '/' + image_path)
            h, w = image.shape[:2]

            # NOTE: random size of watermark and random location
            new_width = randint(min(int(w * 3 / 5), 150), max(int(w * 3 / 5), 150))
            watermark_1 = watermark.copy()
            alpha_1 = alpha.copy()
            watermark_1 = resize(watermark_1, width=new_width)
            alpha_1 = resize(alpha_1, width=new_width)
            (wH, wW) = watermark_1.shape[:2]

            while int(wH / 2) >= h - int(wH / 2) - 1 or int(wW / 2) >= w - int(wW / 2) - 1:
                new_width = randint(30, new_width)
                watermark_1 = resize(watermark_1, width=new_width)
                (wH, wW) = watermark_1.shape[:2]
            y_center = randint(int(wH / 2), h - int(wH / 2) - 1)
            x_center = randint(int(wW / 2), w - int(wW / 2) - 1)

            crop_img = image[y_center - floor(wH / 2):y_center + ceil(wH / 2),
                       x_center - floor(wW / 2):x_center + ceil(wW / 2)]

            h_1, w_1 = crop_img.shape[:2]
            # Giảm độ phân giải sau đó resize lên lại kích thước đúng
            ratio = random.uniform(0.1, 1.0)
            watermark_1 = cv2.resize(watermark_1, (int(w_1 * ratio) + 1, int(h_1 * ratio) + 1))
            alpha_1 = cv2.resize(alpha_1, (int(w_1 * ratio) + 1, int(h_1 * ratio) + 1))
            watermark_1 = cv2.resize(watermark_1, (w_1, h_1))
            alpha_1 = cv2.resize(alpha_1, (w_1, h_1))

            # hệ số opacity, bằng 1 ứng với watermark hiện rõ, bằng 0 ứng với watermark không xuất hiện
            heso = random.uniform(0.5, 0.8)
            alpha_1 = alpha_1.astype(float) / 255
            alpha_1 = alpha_1 * heso
            watermark_1 = watermark_1.astype(float)
            crop_img = crop_img.astype(float)

            watermark_1 = cv2.multiply(alpha_1, watermark_1)
            crop_img = cv2.multiply(1 - alpha_1, crop_img)

            new_image = cv2.add(watermark_1, crop_img)

            # blend the two images together using transparent overlays
            output = image.copy()
            output[y_center - floor(wH / 2):y_center + ceil(wH / 2),
            x_center - floor(wW / 2):x_center + ceil(wW / 2)] = new_image

            # NOTE: write the output image to disk
            x = [0] * 4
            x[1] = y_center - floor(wH / 2)
            x[3] = y_center + ceil(wH / 2)
            x[0] = x_center - floor(wW / 2)
            x[2] = x_center + ceil(wW / 2)
            c1 = max(int(min(x[0], x[2]) - abs(x[2] - x[0]) * 0.25), 0), max(
                int(min(x[1], x[3]) - abs(x[3] - x[1]) * 0.25),
                0)
            c2 = min(int(max(x[0], x[2]) + abs(x[2] - x[0]) * 0.25), image.shape[1]), min(
                int(max(x[1], x[3]) + abs(x[3] - x[1]) * 0.25), image.shape[0])

            image_file_name = format(count, '07d') + ".png"
            label_file_name = format(count, '07d') + ".txt"
            cv2.imwrite(os.path.sep.join((args["output_original"], image_file_name)), image[c1[1]:c2[1], c1[0]:c2[0]])
            p = os.path.sep.join((args["output_image"], image_file_name))
            cv2.imwrite(p, output[c1[1]:c2[1], c1[0]:c2[0]])

            count += 1
        except:
            continue
