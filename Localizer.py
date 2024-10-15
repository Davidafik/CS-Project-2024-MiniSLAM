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
                        
            frame_details = self._miniSlam.process_frame(frame)
                        
            if frame_details is not None:
                R, t = frame_details.R, frame_details.t
                
                c = (-R.T @ t).reshape(3)
                        
                # Calculate the angle on XZ plan - out theta
                # theta = -np.arctan2(R[0,0], R[2,0])
                theta = np.rad2deg(-np.arcsin(-R[2,0]))
                
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

