import cv2
import datetime
from OpenDJI import OpenDJI
import VCS
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from Position import Position
import threading
import numpy as np
import Utils


####################### Input parameters #######################

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_MAP = 'Testing Images/5/map.npy'

# True if you taking the images with the mini 3 pro.
DRONE_CAM = True

# IP address of the connected android device / cv2 video source.
VIDEO_SOURCE = "10.0.0.3"

# Maximum number of images the program will take.
MAX_IMAGES = 100

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

CONTROL = False

################################################################

class Localizer:
    def __init__(self, miniSlam: MiniSLAM, cam) -> None:
        self._position = None
        self._velocity = None
        self._time = None
        
        self._miniSlam = miniSlam
        self._camera = cam
        
        self._live = True

        # Start thread to run in the background
        self._thread = threading.Thread(target=self.__ReadAndProcess__, args=(), name="rtsp_read_thread")
        self._thread.daemon = True
        self._thread.start()
    
    def __ReadAndProcess__(self):
        '''
        read and process frames in the background.
        '''
        while self._live:
            # print(f"pos: {self._position} \nvel: {self._velocity} \ntime: {self._time}")
            ret, frame = self._camera.read()
            time_frame = datetime.datetime.now()
                 
            # If no frame available - skip.
            if not ret:
                print ('Error retriving video stream')
                self._position = None
                self._velocity = None
                self._time = None
                cv2.waitKey(10)
                continue

            # Resize frame.
            frame = cv2.resize(frame, dsize = None,
                                fx = SCALE_READ,
                                fy = SCALE_READ)
                        
            frame_details = self._miniSlam.process_frame(frame)
                        
            if frame_details is not None:
                R, t = frame_details.R, frame_details.t
                
                c = (-R.T @ t).reshape(3)
                        
                # Calculate the angle on XZ plan - out theta
                # theta = -np.arctan2(R[0,0], R[2,0])
                theta = -np.arcsin(-R[2,0])
                
                pos = Position(c, theta)
                if self._position is not None and self._time is not None:
                    self._velocity = (pos - self._position) / (time_frame - self._time).total_seconds()
                self._position = pos
                self._time = time_frame
                
            else:
                print ('Failed to localize frame')
                self._position = None
                self._velocity = None
                self._time = None

    def release(self):
        '''
        Stop the MiniSLAM thread.
        '''
        self._live = False
        self._thread.join()
        
    def getPosition(self):
        if self._position is not None and self._velocity is not None:
            return self._position + self._velocity * (datetime.datetime.now() - self._time).total_seconds()
        return self._position


def put_text(frame, count):
    font = cv2.FONT_HERSHEY_COMPLEX
    
    text = f"{count} frames read"
    frame = cv2.putText(frame, text, (5, 30), font, 1, (153, 153, 0), 1, cv2.LINE_AA)
    
    text = f"Press '{QUIT_KEY}' to quit"
    return cv2.putText(frame, text, (5, 60), font, 1, (153, 153, 0), 1, cv2.LINE_AA)


calib = Calibration(PATH_CALIB)
slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), add_new_pts=False)
slam.load(PATH_MAP)

# print(f"***removing outliers. \n****num points before: {len(mapping._map3d.pts)}")
# mapping.remove_outliers()
# print(f"****num points after: {len(mapping._map3d.pts)}\n")# mapping.remove_isolated_points(0.05)

# Utils.draw_3d_cloud(mapping._map3d.pts)

if DRONE_CAM:
    cam = OpenDJI(VIDEO_SOURCE)
    if CONTROL:
        take_off = ""
        while take_off != "success":
            take_off = cam.takeoff(True)
            print(take_off)
            cv2.waitKey(300)
        cv2.waitKey(5000)
        print(f"enable: {cam.enableControl(get_result=True)}")
else:
    cam = VCS.VideoCapture(VIDEO_SOURCE)


plot_position = Utils.plot_position()
localizer = Localizer(slam, cam)

while cv2.waitKey(WAIT_TIME) != ord(QUIT_KEY):
    ascent, roll, pitch = 0, 0, 0
    pos = localizer.getPosition()
    
    if pos is not None:
        print(f"{pos.getLocVec()}\n")
        c, theta = pos.getLocVec(), pos.getT()
        plot_position.plot_position_heading_new(pos)
        
        ascent, roll, pitch = float(-c[1]), float(-c[0]), float(-c[2])
        pitch = min(0.005, max(-0.005, pitch))
        roll = min(0.005, max(-0.005, roll))
        ascent = min(0.005, max(-0.005, ascent))        
    
    # Send control command.
    if DRONE_CAM and CONTROL:
        print(f'rc {0} {0:.2f} {roll:.2f} {pitch:.2f}')
        print(cam.move(0.,0., roll, pitch, get_result=True))
        # print(cam.move(0, ascent, roll, pitch, get_result=True))
            

    # Display frame.
    if DISPLAY_IMAGE:
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.resize(frame, dsize = None, fx = SCALE_DISPLAY, fy = SCALE_DISPLAY)
        if MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)
        frame = put_text(frame, 0)
        cv2.imshow("frame", frame)

print(cam.move(0, 0, 0, 0, get_result=True))
print(cam.disableControl(True))

localizer.release()

if DISPLAY_IMAGE:
    cv2.destroyAllWindows()

print("*"*70)
# print(f"3d_pts: \n{mapping._map3d.pts}, \nshape {mapping._map3d.pts.shape}\n")
# print(f"ts: \n{ts}, \nshape {ts.shape}\n")

# Utils.draw_3d_cloud(mapping._map3d.pts, ts)


        