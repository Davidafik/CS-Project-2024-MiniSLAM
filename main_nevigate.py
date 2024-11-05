import cv2
import numpy as np
import datetime
import keyboard
from OpenDJI import OpenDJI
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from Localizer import Localizer
from BasicPositionControl import BasicPositionControl as Control
from Position import Position
import FeatureDetector
import Utils


####################### Input parameters #######################

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_MAP = 'Testing Images/3/map.npy'

# IP address of the connected android device.
VIDEO_SOURCE = "10.0.0.3"

# take off and control the drone?
TAKE_OFF = True

# Time to wait between 2 consecutive control iterations (in miliseconds)
WAIT_TIME = 50

# acceptable distace to the target.
OK_ERROR = 0.2

# Scale the image for faster localization.
SCALE_READ = 1

DISPLAY_IMAGE = True

# Scale the image for display:
SCALE_DISPLAY = 0.5

targets = [
    [[ 0.0,  0.0,  0.0],   0],
    [[-2.0,  0.0,  0.0],  10],
    [[-2.0,  0.0,  1.0],  30],
    [[-2.0,  0.0,  2.0],  60],
    [[-2.0,  0.0,  3.0],  90],
    [[-1.0,  0.0,  3.0],  90],
    
]

####################### Initialization #######################

calib = Calibration(PATH_CALIB)

feature_detector = FeatureDetector.FAST_SIFT(threshold=30, max_features=700, nOctaveLayers=5, sigma=1.7)

slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), feature_detector=feature_detector, map_3d_path=PATH_MAP, add_new_pts=False)
# Utils.draw_3d_cloud(slam._map3d.pts)

drone = OpenDJI(VIDEO_SOURCE)
        
pos_plotter = Utils.PlotPosition()

control = Control()
# control.setLookDirection(Position([0, 0, 10]))
    
if TAKE_OFF:
    Utils.take_off(drone)
    cv2.waitKey(3000)
    Utils.enable_control(drone)

localizer = Localizer(slam, scale_image=SCALE_READ)
cv2.waitKey(1000)

    
########################## Main Loop ##########################

# Control loop.
# move the drone along the path.
for target in targets:
    # set the next point in the path as the new target.
    control.setTarget(Position(target[0], target[1]))
    print(f"next target: {target}")

    # keep advancing toward the target until you get small enough error.
    while control.getError() > OK_ERROR and not keyboard.is_pressed('q'):
        ret, frame = drone.read()
        if not ret:
            print ('Error retriving video stream')
            cv2.waitKey(WAIT_TIME)
            continue
        time = datetime.datetime.now()
        curr_pos = localizer.getPosition(frame)
        print(f"Total localization time: {(datetime.datetime.now() - time).total_seconds()} sec.")
        

        print(f"dist to target: {control.getError()}")
        print(f"curr position: {curr_pos}")
        pos_plotter.plot_position_heading_new(curr_pos)

        # Send control command.
        if TAKE_OFF:
            LR,DU,BF,RCW = control.getRCVector(curr_pos)
            print(f'LR: {LR:.3f}\t DU: {DU:.3f}\t BF: {BF:.3f}\t RCW: {RCW:.3f}')
            print(drone.move(RCW, DU, LR, BF, get_result=True))
                
        # Display frame.
        if DISPLAY_IMAGE:
            frame = cv2.resize(frame, dsize = None, fx = SCALE_DISPLAY, fy = SCALE_DISPLAY)
            cv2.imshow("frame", frame)
        
        cv2.waitKey(WAIT_TIME)
        print("*"*70)
    cv2.waitKey(200)
    

if TAKE_OFF:
    # return the control to the remote controller.
    print(f"stop the drone: {drone.move(0, 0, 0, 0, get_result=True)}")
    print(f"land {drone.land(get_result=True)}")
    print(f"disable control: {drone.disableControl(get_result=True)}")

if DISPLAY_IMAGE:
    cv2.destroyAllWindows()

