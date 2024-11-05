import cv2
import datetime
from MiniSLAM import MiniSLAM
from Position import Position
import numpy as np

class Localizer:
    def __init__(self, miniSlam: MiniSLAM, scale_image: float = 1, max_dist_from_prev = 10) -> None:        
        self._miniSlam = miniSlam
        self.scale_image = scale_image
        self._max_dist_from_prev = max_dist_from_prev
        self._prev_t = np.zeros(3)
        
    def getPosition(self, frame: np.ndarray) -> Position:
        # # Resize frame.
        # frame = cv2.resize(frame, dsize = None,
        #                     fx = self.scale_image,
        #                     fy = self.scale_image)
        
        frame_details = self._miniSlam.process_frame(frame)
                    
        if frame_details is None:
            return None
            
        R, t = frame_details.R, frame_details.t
        
        if self._prev_t is not None and np.sum(np.absolute(self._prev_t - t)) > self._max_dist_from_prev:
            return None
        self._prev_t = t

        c = (-R.T @ t).reshape(3)
        
        # flip y axis so it points up.
        c[1] *= -1

        # Calculate the angle on XZ plan
        theta = np.rad2deg(np.arctan2(R[2,0], R[0,0]))
        
        return Position(c, theta)

#test#
if __name__ == "__main__":
    import VCS
    from Calibration import Calibration
    from OpenDJI import OpenDJI #, EventListener
    import FeatureDetector
    import Utils
    import keyboard
    
    ####################### Input Parameters #######################

    PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
    PATH_MAP = 'Testing Images/3/map.npy'

    # True if you taking the images with the mini 3 pro.
    DRONE_CAM = True

    # IP address of the connected android device / cv2 video source.
    VIDEO_SOURCE = "10.0.0.3"

    # Time to wait between 2 consecutive frame savings (in miliseconds)
    WAIT_TIME = 10

    SCALE_READ = 1

    DISPLAY_IMAGE = True

    # Scale the image for display:
    SCALE_DISPLAY = 0.5

    ####################### Initialization #######################
    # gimbleYaw = [0.]
    # class GimbleYawListener(EventListener):
    #     def onValue(self, value: str | None):
    #         gimbleYaw[0] = float(value)
        
    if DRONE_CAM:
        cam = OpenDJI(VIDEO_SOURCE)
        # cam.listen("Gimbal", "YawRelativeToAircraftHeading", GimbleYawListener())
    else:
        cam = VCS.VideoCapture(VIDEO_SOURCE)
            
    calib = Calibration(PATH_CALIB)
    feature_detector = FeatureDetector.FAST_SIFT(threshold=30, max_features=300, nOctaveLayers=5, sigma=1.7)
    slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), feature_detector=feature_detector, map_3d_path=PATH_MAP, add_new_pts=False)

    plot_position = Utils.PlotPosition()

    localizer = Localizer(slam, scale_image=SCALE_READ, max_dist_from_prev = 5)
        
    ########################## Main Loop ##########################

    while not keyboard.is_pressed('q'):
        ret, frame = cam.read()        
        if not ret:
            print ('Error retriving video stream')
            cv2.waitKey(WAIT_TIME)
            continue

        time = datetime.datetime.now()
        curr_pos = localizer.getPosition(frame)
        # if curr_pos is None:
        #     cv2.waitKey(WAIT_TIME)
        #     continue
        # curr_pos._t -= gimbleYaw[0]
        print(f"Total localization time: {(datetime.datetime.now() - time).total_seconds()} sec.")

        # print(gimbleYaw[0])
        print(curr_pos)        
        plot_position.plot_position_heading_new(curr_pos)

        # Display frame.
        if DISPLAY_IMAGE:
            frame = cv2.resize(frame, dsize = None, fx = SCALE_DISPLAY, fy = SCALE_DISPLAY)
            cv2.imshow("frame", frame)
        
        print("*"*70, "\n")
        cv2.waitKey(WAIT_TIME)

    if DISPLAY_IMAGE:
        cv2.destroyAllWindows()
