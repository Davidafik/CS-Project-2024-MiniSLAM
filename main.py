import cv2
import numpy as np

from OpenDJI import OpenDJI
from Mapping import Mapping
from Calibration import Calibration
from FrameDetails import FrameDetails
import Utils

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_IMAGES = 'Testing Images/1'

# PATH_CALIB = "Camera Calibration/CalibDavidLaptop/Calibration.npy"
# PATH_IMAGES = "Testing Images/3"

IMAGE_SCALE = 0.75
SHOW_MATCHES = False

np.set_printoptions(precision=3, suppress=True)

calib = Calibration(PATH_CALIB)
mapping = Mapping(calib.getIntrinsicMatrix(), calib.getExtrinsicMatrix(), add_new_pts=True)
# mapping.load(f"{PATH_IMAGES}/map.npy")

images = Utils.read_images(PATH_IMAGES, IMAGE_SCALE)
Rs, ts = np.empty((0,3,3), float), np.empty((0,3), float)

plot_position = Utils.plot_position()

prevFrameDetails = None
for i, frame in enumerate(images):
    frame_details = mapping.process_frame(frame)
    # Utils.drawKeyPoints(frame, frame_details.kp)
    if frame_details is None:
        continue
    R, t = frame_details.R, frame_details.t
    print(f"{i}:\nR{i}: \n{R}\nt{i}: \n{t}\n")
    plot_position.plot_position_heading(R, t)
    
    if SHOW_MATCHES and i > 0:
        Utils.drawMatches(images[i-1], frame, prevFrameDetails.kp, frame_details.kp, mapping._matches, numDraw = 1000)
    prevFrameDetails = frame_details

    # Rs = np.vstack((Rs, R.reshape((1,3,3))))
    ts = np.vstack((ts, t.T))

mapping.save(f"{PATH_IMAGES}/map.npy")
# mapping.load(f"{PATH_IMAGES}/map.npy")


cv2.destroyAllWindows()

print("*"*50)
print(f"3d_pts: \n{mapping._map3d.pts}, \nshape {mapping._map3d.pts.shape}\n")
print(f"ts: \n{ts}, \nshape {ts.shape}\n")
# print(f"Rs: {Rs}, \nshape {Rs.shape}\n")

# Utils.draw_3d_cloud(mapping._global_3d_pts)
Utils.draw_3d_cloud(mapping._map3d.pts, ts)












