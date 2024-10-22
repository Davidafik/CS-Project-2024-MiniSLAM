import cv2
import keyboard
from OpenDJI import OpenDJI
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from Localizer import Localizer
from BasicPositionControl import BasicPositionControl as Control
from Position import Position
import Utils


####################### Input parameters #######################

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_MAP = 'Testing Images/7/map.npy'

# IP address of the connected android device.
VIDEO_SOURCE = "10.0.0.4"

# take off and control the drone?
TAKE_OFF = True

# Time to wait between 2 consecutive control iterations (in miliseconds)
WAIT_TIME = 50

# acceptable distace to the target.
OK_ERROR = 0.25

# Scale the image for faster localization.
SCALE_READ = 0.8

DISPLAY_IMAGE = False

# Scale the image for display:
SCALE_DISPLAY = 0.5

# pressing this key will close the program.
QUIT_KEY = 'q'


targets = [
    [-1,  0, -1],
    [0,  0, 0],
    # [0,  0,-3],
    # [3,  0,-3],
    # [3,  0, 0],
    # [0,  0, 0],
]

####################### Initialization #######################

drone = OpenDJI(VIDEO_SOURCE)
        
calib = Calibration(PATH_CALIB)

slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), map_3d_path=PATH_MAP, add_new_pts=False)
# Utils.draw_3d_cloud(slam._map3d.pts)

pos_plotter = Utils.PlotPosition()

control = Control()
control.setLookDirection(Position([0, 0, 1e1]))
    
if TAKE_OFF:
    Utils.take_off(drone)
    cv2.waitKey(3000)
    Utils.enable_control(drone)

localizer = Localizer(slam, drone, scale_image=SCALE_READ)
cv2.waitKey(1000)

    
########################## Main Loop ##########################

# Control loop.
# move the drone along the path.
for target in targets:
    # set the next point in the path as the new target.
    control.setTarget(Position(target))

    # keep advancing toward the target until you get small enough error.
    while control.getError() > OK_ERROR and not keyboard.is_pressed(QUIT_KEY):
        print("error: ", control.getError())
        
        curr_pos = localizer.getPosition()
        print(curr_pos)
        
        pos_plotter.plot_position_heading_new(curr_pos)

        # Send control command.
        if TAKE_OFF:
            LR,DU,BF,RCW = control.getRCVector(curr_pos)
            print(f'rc {RCW:.2f} {DU:.2f} {LR:.2f} {BF:.2f}')
            print(drone.move(RCW, DU, LR, BF, get_result=True))
                
        # Display frame.
        if DISPLAY_IMAGE:
            results, frame = drone.read()
            if not results:
                continue
            frame = cv2.resize(frame, dsize = None, fx = SCALE_DISPLAY, fy = SCALE_DISPLAY)
            frame = Utils.put_text(frame, 0, QUIT_KEY)
            cv2.imshow("frame", frame)
        
        cv2.waitKey(WAIT_TIME)
        print("*"*70)

localizer.release()

if TAKE_OFF:
    # return the control to the remote controller.
    print(f"stop: {drone.move(0, 0, 0, 0, get_result=True)}")
    print(f"diable control: {drone.disableControl(get_result=True)}")

if DISPLAY_IMAGE:
    cv2.destroyAllWindows()

