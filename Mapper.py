import numpy as np
from MiniSLAM import MiniSLAM
from Calibration import Calibration
from FrameDetails import FrameDetails
from Localizer import Localizer
from Position import Position
import FeatureDetector
import Utils
np.set_printoptions(precision=3, suppress=True)


PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_IMAGES = 'Testing Images/3'

# PATH_CALIB = "Camera Calibration/CalibDavidLaptop/Calibration.npy"
# PATH_IMAGES = "Testing Images/1"

IMAGE_SCALE = 1

# outliers removing params:
min_neighbors, neighbor_dist, min_dist = 10, 0.6, 0.04

feature_detector = FeatureDetector.FAST_SIFT(threshold=35, max_features=0, nOctaveLayers=4, sigma=1.3)

calib = Calibration(PATH_CALIB)

slam = MiniSLAM(calib.getIntrinsicMatrix(), calib.getDistCoeffs(), feature_detector=feature_detector, add_new_pts=True, max_std_new_pts=5)
# slam.load(f"{PATH_IMAGES}/map.npy")

localizer = Localizer(slam, scale_image=IMAGE_SCALE, max_dist_from_prev = 20)

plot_position = Utils.PlotPosition()

images = Utils.read_images(PATH_IMAGES, IMAGE_SCALE)

ts = np.empty((0,3), float)

for i, frame in enumerate(images, 1):
    print(f"\n{i}:")
    curr_pos = localizer.getPosition(frame)
    print(curr_pos)
    
    if curr_pos is not None:
        plot_position.plot_position_heading_new(curr_pos)
        ts = np.vstack((ts, curr_pos.getLocVec()))
    
    # if i%4 == 0:
    #     print(f"\n***removing outliers. \n****num points before: {len(slam._map3d.pts)}")
    #     slam.remove_outliers(min_neighbors, neighbor_dist, min_dist)
    #     print(f"****num points after: {len(slam._map3d.pts)}\n")
    #     # Utils.draw_3d_cloud(mapping._map3d.pts)

print(f"\n***removing outliers. \n****num points before: {len(slam._map3d.pts)}")
slam.remove_outliers(min_neighbors, neighbor_dist, 0.1)
print(f"****num points after: {len(slam._map3d.pts)}\n")

# slam._map3d.rotate_XZ(-15)
# slam._map3d.rotate_YZ(-10)

slam.save(f"{PATH_IMAGES}/map.npy")

# cv2.destroyAllWindows()

print("*"*50)

# Utils.draw_3d_cloud(slam._map3d.pts)

ts[:, 1] *= -1
Utils.draw_3d_cloud(slam._map3d.pts, ts)


