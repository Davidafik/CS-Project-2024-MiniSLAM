import keyboard
import cv2
from os import mkdir, path
from OpenDJI import OpenDJI


####################### Input parameters #######################

# IP address of the connected android device
IP_ADDR = "10.0.0.10"

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 1

# Save folder
SAVE_PATH = 'Camera Calibration/Calib_mini_3_pro'

# Chess board crosses
CROSS = (6, 4)

# Time to wait between frames (in miliseconds)
WAIT_TIME = 400

# Mirror image for more intuitive display.
# Recommended when holding the board in front of the camera.
MIRROR_DISPLAY = True

# Maximum number of images the program will take.
MAX_IMAGES = 50

################################################################

# Create the folder if it not exist
if not path.isdir(SAVE_PATH):
    mkdir(SAVE_PATH)

# Count number of pictures
count = 0
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:

    # Press 'x' to close the program
    print("Press 'x' to close the program")
    while count < MAX_IMAGES and not keyboard.is_pressed('x'):
        # Get frame from the drone.
        frame = drone.getFrame()

        # What to do when no frame available
        if frame is None:
            print("frame is None!")
            continue
    
        # Resize frame - optional
        frame = cv2.resize(frame, dsize = None,
                           fx = SCALE_FACTOR,
                           fy = SCALE_FACTOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CROSS, None)

        # If found, add object points, image points (after refining them)
        if ret:
            count += 1
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(gray, CROSS, corners2, ret)
            # current_time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
            save_to = f"{SAVE_PATH}/image{count:04}.jpg"
            cv2.imwrite(save_to, frame)
            print (f"{count} saved - {save_to}")
        
        gray = cv2.resize(gray, (720, 480))
        if MIRROR_DISPLAY:
            gray = cv2.flip(gray, 1)
        cv2.imshow("Match", gray)
        cv2.waitKey(WAIT_TIME)
        
    print(f"Collection ended. {count} images saved.")
        