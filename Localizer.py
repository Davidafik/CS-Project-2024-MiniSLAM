import cv2
import datetime
from OpenDJI import OpenDJI
from MiniSLAM import MiniSLAM
from Position import Position
import threading
import numpy as np

class Localizer:
    def __init__(self, miniSlam: MiniSLAM, cam: OpenDJI, scale_image: float = 1, scale_map: float = 1) -> None:
        self._position = None
        self._velocity = None
        self._time = None
        
        self._miniSlam = miniSlam
        self._camera = cam
        
        self.scale_image = scale_image
        self.scale_map = scale_map
        
        self._live = True

        # Start thread to run in the background
        self._thread = threading.Thread(target=self._ReadAndProcess, args=(), name="read_process_thread")
        self._thread.daemon = True
        self._thread.start()
    
    def _ReadAndProcess(self):
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
                                fx = self.scale_image,
                                fy = self.scale_image)
            cv2.imshow("", frame)
            cv2.waitKey(10)
                        
            frame_details = self._miniSlam.process_frame(frame)
                        
            if frame_details is not None:
                R, t = frame_details.R, frame_details.t
                
                c = (-R.T @ t).reshape(3)
                
                if self._position is not None and np.linalg.norm(c - self._position.getLocVec()) > 6:
                    self._position = None
                    self._velocity = None
                    self._time = None
                    return
                
                # Calculate the angle on XZ plan - out theta
                # theta = -np.arctan2(R[0,0], R[2,0])
                theta = np.rad2deg(-np.arcsin(-R[2,0]))
                
                pos = Position(c, theta)
                if self._position is not None:
                    dt = (time_frame - self._time).total_seconds()

                    if self._velocity is not None:
                        pos = pos * 0.8 + (self._position + self._velocity * dt) * 0.2
                        
                    self._velocity = (pos - self._position) / dt
                    
                    print(f"Localization time: {dt} sec.")
                    
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
        # if self._position is not None and self._velocity is not None:
        #     return self._position + self._velocity * (datetime.datetime.now() - self._time).total_seconds() * 0.3
        return self._position

#test#
if __name__ == "__main__":
    import VCS
    from Calibration import Calibration
    import Utils
    import keyboard

    ####################### Input Parameters #######################

    PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
    PATH_MAP = 'Testing Images/5/map.npy'

    # True if you taking the images with the mini 3 pro.
    DRONE_CAM = True

    # IP address of the connected android device / cv2 video source.
    VIDEO_SOURCE = "10.0.0.4"

    # Time to wait between 2 consecutive frame savings (in miliseconds)
    WAIT_TIME = 100

    SCALE_READ = 0.9

    DISPLAY_IMAGE = False

    # Scale the image for display:
    SCALE_DISPLAY = 0.5

    # Mirror the image on display.
    MIRROR_DISPLAY = False

    # pressing this key will close the program.
    QUIT_KEY = 'q'

    ####################### Initialization #######################

    if DRONE_CAM:
        cam = OpenDJI(VIDEO_SOURCE)
    else:
        cam = VCS.VideoCapture(VIDEO_SOURCE)
            
    calib = Calibration(PATH_CALIB)
    slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), map_3d_path=PATH_MAP, add_new_pts=False)

    plot_position = Utils.PlotPosition()

    localizer = Localizer(slam, cam, scale_image=SCALE_READ)
        
    ########################## Main Loop ##########################

    while not keyboard.is_pressed(QUIT_KEY):
        curr_pos = localizer.getPosition()
        print(curr_pos)
        
        plot_position.plot_position_heading_new(curr_pos)

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
        cv2.waitKey(WAIT_TIME)
        
    localizer.release()

    if DISPLAY_IMAGE:
        cv2.destroyAllWindows()
