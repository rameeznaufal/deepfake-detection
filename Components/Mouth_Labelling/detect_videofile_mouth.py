# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
from threading import Thread
from datetime import timedelta
import numpy as np
import sys
import argparse
import imutils
import time
import dlib
import cv2

def print_error(*args, **kwargs):
	    print(*args, file=sys.stderr, **kwargs)

def mouth_aspect_ratio(mouth):
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
	mar = (D + E) / (2.0 * C)
	mar = (A + B) / (F) + (D + E) / (2.0 * C)	
		
	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", default="trump.mp4",
	help="video path input")
args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.485

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# start the video stream thread
print("[INFO] starting video stream thread...")
fvs = cv2.VideoCapture(args["video"])

args["video"].strip("/")
args["video"].strip("\\")

video_name = args["video"].split(".")[0].split("/")[1]

time.sleep(1.0)

frame_width = 640
frame_height = 360

if(fvs.get(cv2.CAP_PROP_FRAME_WIDTH) < 640):
	frame_width = int(fvs.get(cv2.CAP_PROP_FRAME_WIDTH))

if(fvs.get(cv2.CAP_PROP_FRAME_HEIGHT) < 360):
	frame_height = int(fvs.get(cv2.CAP_PROP_FRAME_HEIGHT))

FPS = fvs.get(cv2.CAP_PROP_FPS)
# Define the codec and create VideoWriter object.The output is stored in '[video_name]_output.avi' file.
out = cv2.VideoWriter(f"Output/{video_name}_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
time.sleep(1.0)

# Opening file to write the timestamps for closed mouth frames 
f = open(f"Output/{video_name}_frames.txt", 'w')

frame_count = 0
# loop over frames from the video stream
while fvs.isOpened():
    	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = fvs.read()
	frame_count += 1

	time_stamp = timedelta(seconds = (frame_count/FPS))

	if not ret:
		print_error("Can't recieve frame (Stream end?), Exiting...")
		break

	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#print(len(shape))

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[mStart-1:mEnd]

		#[print(m) for m in mouth]

		#print(len(mouth))

		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)

		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.3f}".format(mar), (30, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			pass
			#cv2.putText(frame, "Mouth is Open!", (30,60),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		else :
			cv2.putText(frame, "Mouth is Closed!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
			f.write(f"[{time_stamp}] - {mar} \n")
	# Write the frame into the file 'output.avi'
	out.write(frame)
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#fvs.stop()
f.close()