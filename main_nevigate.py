import cv2
from OpenDJI import OpenDJI
import VCS
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from Localizer import Localizer
from BasicPositionControl import BasicPositionControl as Control
from Position import Position
import Utils


####################### Input parameters #######################

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_MAP = 'Testing Images/5/map.npy'

# True if you taking the images with the mini 3 pro.
DRONE_CAM = True

# IP address of the connected android device / cv2 video source.
VIDEO_SOURCE = "10.0.0.6"

# # Maximum number of images the program will take.
# MAX_IMAGES = 100

# Time to wait between 2 consecutive frame savings (in miliseconds)
WAIT_TIME = 200

SCALE_READ = 0.7

DISPLAY_IMAGE = False

# Scale the image for display:
SCALE_DISPLAY = 0.5

# Mirror the image on display.
MIRROR_DISPLAY = False

# pressing this key will close the program.
QUIT_KEY = 'q'

CONTROL = True

################################################################

if DRONE_CAM:
    cam = OpenDJI(VIDEO_SOURCE)
else:
    cam = VCS.VideoCapture(VIDEO_SOURCE)
        

calib = Calibration(PATH_CALIB)
slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), map_3d_path=PATH_MAP, add_new_pts=False)
# Utils.draw_3d_cloud(slam._map3d.pts)

plot_position = Utils.Plot_position()
localizer = Localizer(slam, cam, scale_image=SCALE_READ)

if CONTROL:
    control = Control()
    control.setLookDirection(Position([0, 0, 1e4]))
    control.setTarget(Position([0,0,0]))
    
    # take off.
    take_off = ""
    while take_off != "success":
        take_off = cam.takeoff(True)
        print(take_off)
        cv2.waitKey(300)
    cv2.waitKey(5000)
    print(f"enable: {cam.enableControl(get_result=True)}")
        
# Control loop.
while cv2.waitKey(WAIT_TIME) != ord(QUIT_KEY):
    curr_pos = localizer.getPosition()
    print(curr_pos)
    
    plot_position.plot_position_heading_new(curr_pos)
        
    # Send control command.
    if DRONE_CAM and CONTROL:
        LR,DU,BF,RCW = control.getRCVector(curr_pos)
        print(f'rc {RCW:.2f} {DU:.2f} {LR:.2f} {BF:.2f}')
        print(cam.move(RCW, DU, LR, BF, get_result=True))
            
    # Display frame.
    if DISPLAY_IMAGE:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.resize(frame, dsize = None, fx = SCALE_DISPLAY, fy = SCALE_DISPLAY)
        if MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)
        frame = Utils.put_text(frame, 0, QUIT_KEY)
        cv2.imshow("frame", frame)

localizer.release()

if DRONE_CAM and CONTROL:
    print(cam.move(0, 0, 0, 0, get_result=True))
    print(cam.disableControl(True))

if DISPLAY_IMAGE:
    cv2.destroyAllWindows()

print("*"*70)
# print(f"3d_pts: \n{mapping._map3d.pts}, \nshape {mapping._map3d.pts.shape}\n")
# print(f"ts: \n{ts}, \nshape {ts.shape}\n")

# Utils.draw_3d_cloud(mapping._map3d.pts, ts)


