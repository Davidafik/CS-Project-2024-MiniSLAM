import keyboard
import cv2
from os import mkdir, path
from OpenDJI import OpenDJI
import datetime


####################### Input parameters #######################

# IP address of the connected android device
IP_ADDR = "10.0.0.10"

# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 1

# Save folder
SAVE_PATH = 'Testing Images'

# Time to wait between frames (in milliseconds)
WAIT_TIME = 3000

# Mirror image for more intuitive display.
MIRROR_DISPLAY = False

# Maximum number of images the program will take.
MAX_IMAGES = 50

################################################################

# Create the folder if it's not exist.
if not path.isdir(SAVE_PATH):
    mkdir(SAVE_PATH)

# Count number of pictures.
count = 0

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    last_taken_pic = datetime.datetime.now()

    # Press 'x' to close the program.
    print("Press 'x' to close the program")
    while count < MAX_IMAGES and not keyboard.is_pressed('x'):
        # Get frame from the drone.
        frame = drone.getFrame()

        # What to do when no frame available.
        if frame is None:
            print("frame is None!")
            continue
    
        # Resize frame - optional
        frame = cv2.resize(frame, dsize = None,
                           fx = SCALE_FACTOR,
                           fy = SCALE_FACTOR)
        
        # Save frame to folder.
        if last_taken_pic + datetime.timedelta(milliseconds=WAIT_TIME) < datetime.datetime.now():
            count += 1
            save_to = f"{SAVE_PATH}/image{count:05}.jpg"
            cv2.imwrite(save_to, frame)
            print (f"{count} saved - {save_to}")
            last_taken_pic = datetime.datetime.now()
            frame += 70
        
        # Display frame.
        frame = cv2.resize(frame, (720, 480))
        if MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)
        cv2.imshow("frame", frame)
        cv2.waitKey(50)
        
    print(f"Collection ended. {count} images saved.")
        