import cv2
import numpy as np
from time import sleep
import datetime
from VCS import VideoCapture
from os import mkdir, path



####################### Input parameters #######################

webcam = VideoCapture(0)

# Save folder
folder = 'Camera Calibration/Calib1'

# Chess board crosses
cross = (7, 7)


####################### Do not change #######################

# Create the folder if it not exist
if not path.isdir(folder):
    mkdir(folder)

# Count number of pictures
count = 0
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Last time a photo taken
last_taken_pic = datetime.datetime.now()

while True:
    ret, frame = webcam.read()

    for _ in range(40):
        ret, frame = webcam.read()

    if not ret:
        print ('Error retriving video stream')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, cross, None)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(gray, cross, corners2, ret)
    
    img = cv2.resize(gray, (720, 480))
    cv2.imshow("Match", img)

    key = cv2.waitKey(1)

    if ret and last_taken_pic + datetime.timedelta(milliseconds=500) < datetime.datetime.now():
        count = count + 1
        last_taken_pic = datetime.datetime.now()
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
        cv2.imwrite(folder + '/' + str(current_time) + '.jpg', frame)
        print ("saved - " + str(current_time) + " - " + str(count))

    elif key == ord("q"):
        break
    
webcam.release()