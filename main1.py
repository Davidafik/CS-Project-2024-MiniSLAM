import keyboard
import cv2
import numpy as np
from OpenDJI import OpenDJI
import datetime
from Calibration import Calibration
from Mapping import Mapping
import Utils


####################### Input parameters #######################

# IP address of the connected android device
IP_ADDR = "10.0.0.10"

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"


# The image from the drone can be quit big,
#  use this to scale down the image:
SCALE_FACTOR = 1

# Time to wait between frames (in milliseconds)
WAIT_TIME = 1000

# Mirror image for more intuitive display.
MIRROR_DISPLAY = False

# Maximum number of images the program will take.
MAX_IMAGES = 100

SHOW_MATCHES = False

################################################################


np.set_printoptions(precision=3, suppress=True)

calib = Calibration(PATH_CALIB)
mapping = Mapping(calib.getIntrinsicMatrix(), calib.getExtrinsicMatrix())

Rs, ts = np.empty((0,3,3), float), np.empty((0,3), float)

plot_position = Utils.plot_position()

# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    # Count number of pictures.
    i = 0
    
    last_taken_pic = datetime.datetime.now()

    # Press 'x' to close the program.
    print("Press 'x' to close the program")
    while i < MAX_IMAGES and not keyboard.is_pressed('x'):
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
            
            frame_details = mapping.process_frame(frame)
            if frame_details is not None:
                i += 1
                R, t = frame_details.R, frame_details.t
                print(f"{i}:\nR{i}: \n{R}\nt{i}: \n{t}\n")
                plot_position.plot_position_heading(R, t)  
                # Rs = np.vstack((Rs, R.reshape((1,3,3))))
                ts = np.vstack((ts, t.T))

            last_taken_pic = datetime.datetime.now()
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=70)
            
        
        # Display frame.
        frame = cv2.resize(frame, (720, 480))
        if MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)
        cv2.imshow("frame", frame)
        cv2.waitKey(50)
        
    cv2.destroyAllWindows()
    
    print("*"*50)
    # print(f"3d_pts: \n{mapping._points_3d_pts}, \nshape {mapping._points_3d_pts.shape}\n")
    print(f"ts: \n{ts}, \nshape {ts.shape}\n")
    # print(f"Rs: {Rs}, \nshape {Rs.shape}\n")

    # Utils.draw_3d_cloud(mapping._global_3d_pts)
    Utils.draw_3d_cloud(mapping._map3d.pts, ts)


        