import cv2
import numpy as np
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from FrameDetails import FrameDetails
import Utils
from Position import Position

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_IMAGES = 'Testing Images/8'

# PATH_CALIB = "Camera Calibration/CalibDavidLaptop/Calibration.npy"
# PATH_IMAGES = "Testing Images/1"

IMAGE_SCALE = 1

# outliers removing params:
min_neighbors, neighbor_dist, min_dist = 10, 0.6, 0.04

SHOW_MATCHES = False

np.set_printoptions(precision=3, suppress=True)

calib = Calibration(PATH_CALIB)
slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), add_new_pts=True, max_std_new_pts=5)
# mapping.load(f"{PATH_IMAGES}/map.npy")

images = Utils.read_images(PATH_IMAGES, IMAGE_SCALE)
Rs, ts = np.empty((0,3,3), float), np.empty((0,3), float)

plot_position = Utils.PlotPosition()

prevFrameDetails = None
for i, frame in enumerate(images, 1):
    print(f"{i}:")
    frame_details = slam.process_frame(frame)
    # Utils.drawKeyPoints(frame, frame_details.kp)
    if frame_details is None:
        continue
    R, t = frame_details.R, frame_details.t
    
    theta = np.rad2deg(np.arctan2(R[2,0], R[0,0]))
    print(f"theta: {theta}")
    # print(f"R{i}: \n{R}\n")
    print(f"t{i}: {t.reshape(3)}\n")
    plot_position.plot_position_heading(R, t)
    
    if SHOW_MATCHES and i > 0:
        Utils.drawMatches(images[i-1], frame, prevFrameDetails.kp, frame_details.kp, slam._matches, numDraw = 500)
    prevFrameDetails = frame_details

    # Rs = np.vstack((Rs, R.reshape((1,3,3))))
    ts = np.vstack((ts, t.T))
    if i%4 == 0:
        print(f"***removing outliers. \n****num points before: {len(slam._map3d.pts)}")
        slam.remove_outliers(min_neighbors, neighbor_dist, min_dist)
        print(f"****num points after: {len(slam._map3d.pts)}\n")
        # Utils.draw_3d_cloud(mapping._map3d.pts)

print(f"***removing outliers. \n****num points before: {len(slam._map3d.pts)}")
slam.remove_outliers(min_neighbors, neighbor_dist, 0.1)
print(f"****num points after: {len(slam._map3d.pts)}\n")

# slam._map3d.rotate_XZ(-15)
slam._map3d.rotate_YZ(-10)

slam.save(f"{PATH_IMAGES}/map.npy")
# mapping.load(f"{PATH_IMAGES}/map.npy")


# cv2.destroyAllWindows()

print("*"*50)
print(f"3d_pts shape: {slam._map3d.pts.shape}\n")
# print(f"ts: \n{ts}, \nshape {ts.shape}\n")
# print(f"Rs: {Rs}, \nshape {Rs.shape}\n")

Utils.draw_3d_cloud(slam._map3d.pts)
# Utils.draw_3d_cloud(slam._map3d.pts, ts)


