import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from Mapping import Mapping
from Calibration import Calibration
from FrameDetails import FrameDetails

np.set_printoptions(precision=3, suppress=True)

def read_images(folder_path: str, size : float = 0.25):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only .jpg files and sort them by their numeric index
    image_files = sorted([f for f in files if f.endswith('.jpg')],
                         key=lambda x: int(x.split('image')[-1].split('.jpg')[0]))
    
    # Initialize an empty list to store the images
    images = []
    
    # Loop over the sorted image files and read each image
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        image = cv2.imread(img_path)
            
        if image is not None:  # Ensure the image was successfully read
            image = cv2.resize(image, dsize=None, fx=size, fy=size)
            # image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        else:
            print(f"Warning: {img_path} could not be read.")
    
    return images

def plot2images(image1 : np.ndarray, image2 : np.ndarray, title: str = '', figsize: tuple = (12, 4)) -> None:
    """
    Plots two images using matplotlib.

    Args:
        image1 (np.ndarray): The left image. 
        image2 (np.ndarray): The right image.
        title (str, optional): The title of the plot. Defaults to ''.
        figsize (tuple, optional): The size of the plot. Defaults to (12, 4).
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].imshow(image1)
    ax[0].set_title('Image 1')
    ax[0].axis('off')
    
    ax[1].imshow(image2)
    ax[1].set_title('Image 2')
    ax[1].axis('off')
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def plotImage(image : np.ndarray, title: str = '', figsize: tuple = (12, 4)) -> None:
    """
    Plots an image using matplotlib.

    Args:
        image (np.ndarray): The image to plot. 
        title (str, optional): The title of the plot. Defaults to ''.
        figsize (tuple, optional): The size of the plot. Defaults to (12, 4).
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def drawKeyPoints(image : np.ndarray, kp : np.ndarray) -> None:
    """
    Draws Key points on the an image and plots it. 

    Args:
        image1 (np.ndarray): The image.
        kp1 (np.ndarray): Key points for the image
    """
    im = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(im)
    plt.title('Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def drawMatches(image1 : np.ndarray, image2 : np.ndarray, kp1 : np.ndarray, kp2 : np.ndarray, matches : np.ndarray, numDraw : int = 50) -> None:
    """
    1: Draws the points that has a match on the two images and plots them.
    2: Draws lines between points that matched.

    Args:
        image1 (np.ndarray): The left image.
        image2 (np.ndarray): The right image.
        kp1 (np.ndarray): The Key points of the left image.
        kp2 (np.ndarray): The Key points of the right image.
        matches (np.ndarray): The matches between the Key points.
        numDraw (int, optional): The number of lines to draw. Defaults to 50.
    """
    # draw connecting lines between matches
    imageMatches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:numDraw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    imageMatches = cv2.resize(imageMatches, dsize=None, fx=0.25, fy=0.25)
    cv2.imshow("matches", imageMatches)
    while cv2.waitKey(100) != ord('q'):
        pass

def draw_3d_cloud(points, cam_t : np.ndarray = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s = 2)
    
    # Equalizing the axies
    axis_max = np.max(points, axis = 0)
    axis_min = np.min(points, axis = 0)

    if cam_t is not None:
        ax.scatter(cam_t[:, 0], cam_t[:, 1], cam_t[:, 2], s = 2, marker="s")
        axis_max = np.max(np.vstack((np.max(cam_t, axis = 0), axis_max)), axis=0)
        axis_min = np.min(np.vstack((np.min(cam_t, axis = 0), axis_min)), axis=0)
        
    axis_mid = (axis_max + axis_min) / 2
    axis_rng = np.max(axis_max - axis_mid)
    
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlim3d([axis_mid[0] - axis_rng, axis_mid[0] + axis_rng])
    ax.set_ylim3d([axis_mid[1] - axis_rng, axis_mid[1] + axis_rng])
    ax.set_zlim3d([axis_mid[2] - axis_rng, axis_mid[2] + axis_rng])
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.title('3D Point Cloud')
    plt.show()

PATH_CALIB = "Camera Calibration/CalibMini3Pro/Calibration.npy"
PATH_IMAGES = 'Testing Images'
IMAGE_SCALE = 1
SHOW_MATCHES = True

calib = Calibration(PATH_CALIB)
mapping = Mapping(calib.getIntrinsicMatrix(), calib.getExtrinsicMatrix())

images = read_images(PATH_IMAGES, IMAGE_SCALE)
Rs, ts = np.empty((0,3,3), float), np.empty((0,3), float)

prevFrameDetails = None
for i, frame in enumerate(images):
    # print(i)
    frame_details = mapping.process_frame(frame)
    # drawKeyPoints(image, frame_details.key_points)
    if frame_details is None:
        continue
    R, t = frame_details.R, frame_details.t
    print(f"R{i}: \n{R}\nt{i}: \n{t}\n")
    
    if SHOW_MATCHES and i > 0:
        drawMatches(images[i-1], frame, prevFrameDetails.key_points, frame_details.key_points, mapping._matches, numDraw = 1000)
    prevFrameDetails = frame_details

    # Rs = np.vstack((Rs, R.reshape((1,3,3))))
    ts = np.vstack((ts, t.T))

cv2.destroyAllWindows()

print("******************************************************")
# print(f"3d_pts: \n{mapping._points_3d_pts}, \nshape {mapping._points_3d_pts.shape}\n")
print(f"ts: \n{ts}, \nshape {ts.shape}\n")
# print(f"Rs: {Rs}, \nshape {Rs.shape}\n")

draw_3d_cloud(mapping._global_3d_pts)
draw_3d_cloud(mapping._global_3d_pts, ts)












