import cv2
import datetime
from os import mkdir, path
from OpenDJI import OpenDJI
import VCS


####################### Input parameters #######################

# True if you taking the images with the mini 3 pro.
DRONE_CAM = True

# IP address of the connected android device
VIDEO_SOURCE = "10.0.0.10"

# Save folder
SAVE_PATH = 'Testing Images/1'

# Maximum number of images the program will take.
MAX_IMAGES = 100

# Time to wait between 2 consecutive frame savings (in miliseconds)
WAIT_TIME = 750

# Scale the image for display:
SCALE_FACTOR = 0.5

# Mirror the image on display.
MIRROR_DISPLAY = False

# pressing this key will close the program.
QUIT_KEY = 'q'

################################################################

def put_text(frame, count):
    font = cv2.FONT_HERSHEY_COMPLEX
    
    text = f"{count} frames saved"
    frame = cv2.putText(frame, text, (5, 30), font, 1, (153, 153, 0), 1, cv2.LINE_AA)
    
    text = f"Press '{QUIT_KEY}' to quit"
    return cv2.putText(frame, text, (5, 60), font, 1, (153, 153, 0), 1, cv2.LINE_AA)
    


# Create the folder if it does not exist.
if not path.isdir(SAVE_PATH):
    mkdir(SAVE_PATH)

if DRONE_CAM:
    cam = OpenDJI(VIDEO_SOURCE)
else:
    cam = VCS.VideoCapture(VIDEO_SOURCE)


# Count the number of frames.
count = 0

# Time of the last saved frame.
last_saved_frame = datetime.datetime.now()

# Press QUIT_KEY to close the program.
while count < MAX_IMAGES and cv2.waitKey(20) != ord(QUIT_KEY):
    # Get frame from the camera.
    ret, frame = cam.read()

    # What to do when no frame available.
    if not ret:
        print ('Error retriving video stream')
        continue
    
    # Save frame to folder every WAIT_TIME ms.
    if last_saved_frame + datetime.timedelta(milliseconds=WAIT_TIME) < datetime.datetime.now():
        count += 1
        last_saved_frame = datetime.datetime.now()
        
        # Save frame to folder. 
        save_to = f"{SAVE_PATH}/image{count:04}.jpg"
        cv2.imwrite(save_to, frame)
        print (f"{count} saved - {save_to}")

        # Change the frame's brightness to indicate with a flase to the user that a frame was saved.
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=50)
            
    # Display frame.
    frame = cv2.resize(frame, dsize = None, fx = SCALE_FACTOR, fy = SCALE_FACTOR)
    if MIRROR_DISPLAY:
        frame = cv2.flip(frame, 1)
    frame = put_text(frame, count)
    cv2.imshow("frame", frame)
    
cv2.destroyAllWindows()
print(f"Collection ended. {count} images saved.")
      