# import the necessary packages
# MAR Threshold is 0.485

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# frams_path is the path to the directory containing all the captured frames
def calculate_mar(mouth):
    	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[13], mouth[19]) # 62, 68
	B = dist.euclidean(mouth[15], mouth[17]) # 64, 66
	D = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	E = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
	F = dist.euclidean(mouth[12], mouth[16]) # 61, 65


	# compute the mouth aspect ratio
	mar = (A + B) / (F) + (D + E) / (2.0 * C)	
		
	# return the mouth aspect ratio
	return mar



def mouth_aspect_ratio(frames_path):
    # define one constants, for mouth aspect ratio to indicate open mouth
    mar = []

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print(os.getcwd())
    print("[INFO] loading facial landmark predictor...")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./Components/Mouth_Labelling/shape_predictor_68_face_landmarks.dat")

    for filename in os.listdir(frames_path):
        # Check if the file is an image (ends with .jpg or .png)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct the full image path
            image_path = os.path.join(frames_path, filename)

            # grab the indexes of the facial landmarks for the mouth
            (mStart, mEnd) = (49, 68)

            # read the image and convert it to grayscale
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the mouth coordinates, then use the
                # coordinates to compute the mouth aspect ratio
                mouth = shape[mStart-1:mEnd]
                mar.append([filename, calculate_mar(mouth)])

    return mar
