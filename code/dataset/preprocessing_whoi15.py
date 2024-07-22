
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import copy
import numpy as np
import albumentations as A
import cv2

from code.dataset.whoi15_padded import PLANKTON_WHOI15_PATH, PLANKTON_PATH


FLAGS = flags.FLAGS
flags.DEFINE_list("classes", ["Asterionellopsis", "Chaetoceros",
                              "Cylindrotheca", "Dactyliosolen", "detritus",
                              "Dinobryon", "Ditylum", "Licmophora",
                              "pennate", "Phaeocystis", "Pleurosigma",
                              "Pseudonitzschia", "Rhizosolenia", "Skeletonema", "Thalassiosira"], "Name of the classes")

flags.DEFINE_string("output_directory", "whoi15_padded_224", "Output directory of processed images.")
flags.DEFINE_string("output_year", "2008", "Output directory of processed images.")

flags.DEFINE_integer("output_size", 224, "Width and Height of processed images.")


def compute_mask(img):
    img1 = cv2.medianBlur(img, 7, img)

    try:

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                     15, 2)

        img2 = cv2.bitwise_not(img2)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    except Exception as e:
        print("0: ", e)

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        img = cv2.bitwise_not(img)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        img = cv2.convertScaleAbs(img)
        im3 = copy.copy(img)
        img = cv2.bitwise_or(img, img2)

        # edge detection
        # im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        z = 0
        frame4 = img
        imy = np.zeros_like(im3)
        ar_max = 0
        ar_max2 = 0
        for i in x:

            #print(x[0].shape)
            # minimum rectangle containing the object
            PO2 = cv2.boundingRect(i)
            area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
            # LET'S CHOOSE THE AREA THAT WE WANT
            #if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

            moments = cv2.moments(i)
            #cv2.drawContours(imy, [i], -1, (255, 0, 0), -1)

            ar = moments['m00']
            if ar > ar_max2:
                ar_max = i
                ar_max2 = ar


        try:

            #j = ar_max
            #PO2 = cv2.boundingRect(i)
            imx = np.zeros_like(im3)

            cv2.drawContours(imx, [ar_max], -1, (255, 0, 0), -1)
            o = imx
            ttemp = copy.copy(im3)
            ttemp[ttemp != 0] = 1
            o[o != 0] = 1
            imy[imy != 0] = 1
            if np.sum(o) < np.sum(imy) * 0.5:
                ############################################
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                # im2, x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                x, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                z = 0
                frame4 = img

                ar_max = 0
                ar_max2 = 0
                for i in x:
                    #print(x[0].shape)

                    # minimum rectangle containing the object
                    PO2 = cv2.boundingRect(i)
                    area = cv2.countNonZero(
                        frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                    # LET'S CHOOSE THE AREA THAT WE WANT
                    if area > 200 and area < np.shape(img)[0] * np.shape(img)[1]:

                        moments = cv2.moments(i)
                        ar = moments['m00']
                        if ar > ar_max2:
                            ar_max = i
                            ar_max2 = ar

                try:
                    i = ar_max
                    #PO2 = cv2.boundingRect(ar_max)
                    optim_contour = np.zeros_like(im3)
                    #area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
                    cv2.drawContours(optim_contour, [ar_max], -1, (255, 255, 0), -1)
                    #######################display objects###########################ll
                    # if np.sum(optim_contour) > 255 * np.shape(optim_contour)[1] * np.shape(optim_contour)[0] - 500:
                    # continue

                    # cv2.imwrite(output_segmentation_train + '/' + files[j] + '/' + files2[aa],optim_contour)

                    # continue

                    return optim_contour
                except Exception as e:
                    print("1: ", e)

            ############################################

            #area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
            optim_contour = np.zeros_like(im3)
            #area = cv2.countNonZero(frame4[PO2[1]:(PO2[1] + PO2[3]), PO2[0]:(PO2[0] + PO2[2])])
            cv2.drawContours(optim_contour, [ar_max], -1, (255, 0, 255), -1)

            #######################display objects###########################ll

            # if np.sum(optim_contour) > 255 * np.shape(optim_contour)[1] * np.shape(optim_contour)[0] - 500:
            # continue

            # cv2.imwrite(output_segmentation_train + '/' + files[j] + '/' + files2[aa],optim_contour)
            return optim_contour
        except Exception as e:
            print("2: ", e)

    except Exception as e:
        print("3: ", e)

    #return optim_contour

"""
def compute_mask(img):
    try:
        # Pre-processing
        img1 = cv2.medianBlur(img, 7, img)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        img2 = cv2.bitwise_not(img2)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Thresholding
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        img = cv2.bitwise_not(img)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        img = cv2.convertScaleAbs(img)

        # Contour detection
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200 and area < img.shape[0] * img.shape[1]:
                moments = cv2.moments(contour)
                if moments['m00'] > max_area:
                    max_contour = contour
                    max_area = moments['m00']

        # Draw contour
        if max_contour is not None:
            output = np.zeros_like(img)
            cv2.drawContours(output, [max_contour], -1, (255, 0, 255), -1)
            return output

    except Exception as e:
        print("Error:", e)
        return None
"""







def get_max_width_height(directory):
    # first bigger image in dir
    max_width = 0
    max_height = 0

    images = os.listdir(directory)

    for file in images:
        img_path = os.path.join(directory , file)

        image = cv2.imread(img_path)

        h, w, _ = image.shape

        if w > max_width and h > max_height:
            max_width = w
            max_height = h

    return max_width, max_height


def preprocess_class(output_directory, split, c, transform):

    # process the original image
    output_image_directory = os.path.join(output_directory, c)
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)

    # list files in img directory
    original_image_directory = os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split,c)

    images = os.listdir(original_image_directory)

    for file in images:
        # make sure file is an image
        if file.endswith('.png'):
            img_path = os.path.join(original_image_directory , file)

            image = cv2.imread(img_path)

            augmented = transform(image=image)
            image_padded = augmented['image']

            cv2.imwrite(os.path.join(output_image_directory, file), image_padded)



def mask_class(output_directory, split, c, transform):

    # process the original image
    output_image_directory = os.path.join(output_directory, c)
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)

    # list files in img directory
    original_image_directory = os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split,c)

    images = os.listdir(original_image_directory)

    for file in images:
        # make sure file is an image
        if file.endswith('.png'):
            img_path = os.path.join(original_image_directory , file)

            image = cv2.imread(img_path)

            print("Computing mask of ", img_path)
            mask = compute_mask(image)

            augmented = transform(image=image, mask=mask)
            mask_padded = augmented['mask']

            print("Mask saved in ", os.path.join(output_image_directory, file))
            cv2.imwrite(os.path.join(output_image_directory, file), mask_padded)


def generate_masks(split="train"):

    split_out = split + "_BIN"
    # now create processed dataset with selected classes
    output_directory = os.path.join(PLANKTON_PATH, FLAGS.output_directory, FLAGS.output_year, split_out)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    for c in FLAGS.classes:
        max_width, max_height = get_max_width_height(os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split, c))



        # trasform images and masks
        pad_if_needed = A.Compose([A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode= cv2.BORDER_CONSTANT, value =0, mask_value=0,  p=1), # value =(100, 100, 100)
                               A.Resize(FLAGS.output_size, FLAGS.output_size, p=1)], is_check_shapes=False, p=1)

        mask_class(output_directory, split, c, pad_if_needed)



def preprocess_split(split="train"):

    # now create processed dataset with selected classes
    output_directory = os.path.join(PLANKTON_PATH, FLAGS.output_directory, FLAGS.output_year, split)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    for c in FLAGS.classes:
        max_width, max_height = get_max_width_height(os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split, c))



        # trasform images and masks
        pad_if_needed = A.Compose([A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode= cv2.BORDER_CONSTANT, value =0, mask_value=0,  p=1), # value =(100, 100, 100)
                               A.Resize(FLAGS.output_size, FLAGS.output_size, p=1)], is_check_shapes=False, p=1)

        preprocess_class(output_directory, split, c, pad_if_needed)


def main(unused_args):
    preprocess_split("train")
    #preprocess_split("TEST_IMAGE")

def main_mask(unused_args):
    generate_masks("train")


"""
if __name__ == "__main__":
    app.run(main)
"""
#"""
if __name__ == "__main__":

    print("Generate masks!")
    app.run(main_mask)
    
#"""