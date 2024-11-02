import numpy as np
import datetime
import copy
import cv2

from FrameDetails import FrameDetails
from Map3D import Map3D
import FeatureDetector

class MiniSLAM:
    """
    _summary_
    """
    def __init__(self, 
                 K: np.ndarray, 
                 distCoeffs: np.ndarray = np.zeros(5), 
                 map_3d_path: str = None, 
                 feature_detector: FeatureDetector.FeatureDetector = FeatureDetector.FAST_SIFT(), 
                 add_new_pts: bool = True, 
                 max_std_new_pts: float = 2, 
                 triangulation_inliers_threshold: float = 7e-3) -> None:
        """
        _summary_

        Args:
            K (np.ndarray): intrinsic matrix of the camera.
            distCoeffs (np.ndarray, optional): distortion coeffitions for the camera. Defaults to np.zeros(5).
            map_3d_path (str, optional): path for a saved 3d map to load on initialization. Defaults to None.
            feature_detector (FeatureDetector.FeatureDetector, optional): which feature detector to use. options in FeatureDetector.py. Defaults to FeatureDetector.FAST_SIFT().
            add_new_pts (bool, optional): whether to add new 3d points to the map. Defaults to True.
            max_std_new_pts (float, optional): standard deviations for filtering new 3d points. Defaults to 2.
            triangulation_inliers_threshold (float, optional): threshold to determine which mathes to the prev frame are good for triangulatoin. Defaults to 7e-3.
        """
        self._K = K
        self._K_inv = np.linalg.inv(K)
        self._distCoeffs = distCoeffs
        self._feature_detector = feature_detector
        
        self._matcher = cv2.BFMatcher()
        
        self._map3d = Map3D(self._feature_detector.emptyDsc(), map_3d_path)
        self._frame_details_prev = None
        
        self._triangulation_inliers_threshold = triangulation_inliers_threshold
        self._max_std_new_pts = max_std_new_pts
        self.add_new_pts = add_new_pts
        
    def load(self, map3dPath: str):
        self._map3d.load(map3dPath)
        self._frame_details_prev = None
                
    def save(self, map3dPath: str):
        self._map3d.save(map3dPath)

    def process_frame(self, new_frame : np.ndarray) -> FrameDetails:
        """
        Processes a new frame by detecting keypoints, computing descriptors, and updating the map
        with newly triangulated 3D points based on matches with previous frames.

        Args:
            new_frame (np.ndarray): The new frame to be processed, which should be an RGB image.

        Returns:
            FrameDetails: An object containing the keypoints, descriptors, and the pose (R, t, P matrix) of the current frame.
                          Returns None if the frame cannot be processed (e.g., localization fails).
        """
        gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        
        time = datetime.datetime.now()
        kp, dsc = self._feature_detector.detectAndCompute(gray_new_frame)
        print(f"detect and comput: {(datetime.datetime.now() - time).total_seconds()} sec.")

        frame_details_curr = FrameDetails(key_points=kp, descriptors=dsc)
        
        # If we have already have 3d points cloud, attempt localization with PnP, and triangulate new 3d points.
        if not self._map3d.isEmpty():
            
            ##### Part 1: Attempt to localize the current frame using PnP (Perspective-n-Point) with the global 3D map. #####
            matches_3d = self._localize_with_PnP(frame_details_curr)
            
            if matches_3d is None:
                return None
                        
            ##### Part 2: Triangulate new points based on matched features to the previous frame: #####

            if self._frame_details_prev is None or not self.add_new_pts:
                # Store the frame for matching with the next one.
                self._frame_details_prev = frame_details_curr
                return frame_details_curr
            
            # Check if the camera movement (based on translation vector) between the current and previous frames is minimal.
            # If the camera hasn't moved significantly, skip triangulation and return the current frame details.
            dist_to_prev = np.linalg.norm(self._frame_details_prev.t - frame_details_curr.t)
            if dist_to_prev < 0.5 or dist_to_prev > 15:
                print(f"no triangulation - camera movement is {dist_to_prev}")
                return frame_details_curr
            
            matches_prev_curr = self._matches_to_prev_frame(frame_details_curr)

            # Filter matches between the previous and current frame that correspond to unknown (new) 3D points
            new_matches_prev_curr = self._filter_matches_new_pts(frame_details_curr, matches_prev_curr, matches_3d)

            pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, new_matches_prev_curr)
                        
            self._triangulate(self._frame_details_prev.P, frame_details_curr.P, pts_2d_prev, pts_2d_curr, new_matches_prev_curr, dsc)

            self._frame_details_prev = frame_details_curr

            return frame_details_curr
        
        # If this is the second frame being processed, attempt to localize using feature matches between the first and the current frames.
        elif self._frame_details_prev is not None:
            
            ##### Part 1: Attempt to localize the current frame using feature matches between the first current frame. #####
            pts_2d_first, pts_2d_curr, matches_prev_curr = self._localize_with_prev_frame(frame_details_curr)
            
            if matches_prev_curr is None:
                return None
            
            ##### Part 2: Triangulate 3d points based on the matched features found in part 1: #####
            if not self.add_new_pts:
                return frame_details_curr
            
            ret = self._triangulate(self._frame_details_prev.P, frame_details_curr.P, pts_2d_first, pts_2d_curr, matches_prev_curr, dsc)
            
            if not ret:
                return None
            
            self._frame_details_prev = frame_details_curr
            
            return frame_details_curr

        # If this is the very first frame being processed, initialize its pose as the identity.
        else:
            frame_details_curr.R = np.eye(3)
            frame_details_curr.t = np.zeros((3, 1))
            frame_details_curr.P = np.hstack((self._K, np.zeros((3, 1))))

            self._frame_details_prev = frame_details_curr

            return frame_details_curr
        
    def _localize_with_PnP(self, frame_details_curr: FrameDetails) -> np.ndarray:
        """
        Solve the Perspective-n-Point (PnP) problem to estimate the camera's rotation and translation
        based on 2D-3D correspondences between the current frame and the global 3D map.
        Save Rotation matrix, translation vector and projection matrix in frame_details_curr.
        
        Args:
            frame_details_curr (FrameDetails):
                The FrameDetails object containing the keypoints and descriptors for the current frame.
        
        Returns:
            matches_3d_curr (np.ndarray):
                A numpy array of DMatch objects representing the 2D-3D correspondences between the current frame and global 3D points.
                Returns None if there are insufficient matches or if PnP fails.
        """
        
        time = datetime.datetime.now()
        matches_3d_curr = self._find_matches(self._map3d.dsc, frame_details_curr.dsc)
        print(f"find mathes to 3d map: {(datetime.datetime.now() - time).total_seconds()} sec.")
        
        # print(f"num matches: {len(matches_3d_curr)}")
        
        # At least 5 matches are required for solving PnP.
        if len(matches_3d_curr) < 5:
            print("Not enough matches!")
            return None
                
        pts_3d = np.array([self._map3d.pts[m.queryIdx] for m in matches_3d_curr])
        pts_2d_curr = np.array([frame_details_curr.kp[m.trainIdx].pt for m in matches_3d_curr], dtype=np.float64)
                
        if self._frame_details_prev is not None:
            rvec_prev, tvec_prev = cv2.Rodrigues(self._frame_details_prev.R)[0], copy.deepcopy(self._frame_details_prev.t)
        else:
             rvec_prev, tvec_prev = None, None
        
        time = datetime.datetime.now()
        try:
            success, rvec, t, _ = cv2.solvePnPRansac(
                objectPoints=pts_3d,            # 3D points in the global map.
                imagePoints=pts_2d_curr,        # Corresponding 2D points in the current frame.
                cameraMatrix=self._K,           # Intrinsic camera matrix.
                distCoeffs=self._distCoeffs,    # Distortion coefficients of the camera.
                flags=cv2.SOLVEPNP_ITERATIVE,   # Method for solving PnP.
                # confidence=0.9,                 # Confidence level for RANSAC.
                # reprojectionError=0.99,         # Maximum allowed reprojection error.
                rvec=rvec_prev,                 # Initial guess for rotation - the previous frame's rotation.
                tvec=tvec_prev,                 # Initial guess for translation - the previous frame's translation.
                useExtrinsicGuess=rvec_prev is not None,    # Use the given guess.
                # iterationsCount=200             # Number of RANSAC iterations.
            )
        
        except ValueError:
            print(f"PnP: {(datetime.datetime.now() - time).total_seconds()} sec.")
            print(f"PnP error! {ValueError}")
            return None

        if not success:
            print(f"PnP: {(datetime.datetime.now() - time).total_seconds()} sec.")
            print("PnP failed!")
            return None
        
        print(f"PnP: {(datetime.datetime.now() - time).total_seconds()} sec.")
        
        R, _ = cv2.Rodrigues(rvec)
        P = self._K @ np.hstack((R, t))

        frame_details_curr.R = R
        frame_details_curr.t = t
        frame_details_curr.P = P

        # Return the 2D-3D matches found between the global map and the current frame.
        return matches_3d_curr

    def _localize_with_prev_frame(self, frame_details_curr: FrameDetails) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the pose of the current frame relative to the previous frame by matching features and
        computing the essential matrix.
        Save Rotation matrix, translation vector and projection matrix in frame_details_curr.

        Args:
            frame_details_prev (FrameDetails):
                The FrameDetails object containing keypoints, descriptors, and pose of the previous frame.
            
            frame_details_curr (FrameDetails):
                The FrameDetails object containing keypoints and descriptors of the current frame.

        Returns:
            pts_2d_prev (np.ndarray): The 2D points from the previous frame that match the current frame's keypoints.
            
            pts_2d_curr (np.ndarray): The 2D points from the current frame that match the previous frame's keypoints.
            
            matches (np.ndarray): A numpy array of DMatch objects representing the matches between the previous and current frame.
                
            #### Returns None, None, None if localization  failed.
            
        """
        time = datetime.datetime.now()
        matches = self._find_matches(self._frame_details_prev.dsc, frame_details_curr.dsc)
        print(f"find matches to prev: {(datetime.datetime.now() - time).total_seconds()} sec.")
        
        if len(matches) < 5:
            print("Not enough matches!")
            return None, None, None

        time = datetime.datetime.now()
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches)
        E, matches = self._find_essntial(pts_2d_prev, pts_2d_curr, matches)
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches)
        P, R, t = self._calcPosition(E, pts_2d_prev, pts_2d_curr)
        print(f"localization with prev: {(datetime.datetime.now() - time).total_seconds()} sec.")
        
        frame_details_curr.R = R
        frame_details_curr.t = t
        frame_details_curr.P = P

        # Return the matched 2D points from the previous and current frames, as well as the refined matches.
        return pts_2d_prev, pts_2d_curr, matches

    def _calcPosition(self, E: np.ndarray, pts_2d_1: np.ndarray, pts_2d_2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover the camera's pose (rotation and translation) from the essential matrix and corresponding 2D points.

        Args:
            E (np.ndarray):
                The 3x3 essential matrix, encoding the epipolar geometry between two views.
            
            pts_2d_1 (np.ndarray):
                An array of 2D points from the first image (previous frame).
            
            pts_2d_2 (np.ndarray):
                An array of 2D points from the second image (current frame).
            
            normalization (bool): 
                If True, normalizes the translation vector `t` to unit length. 
                [optional, Default is True].

        Returns:
            P (np.ndarray): The 3x4 projection matrix for the current frame, which is a combination of the camera's intrinsic matrix `K`
                and the estimated rotation matrix `R` and translation vector `t`.
            
            R (np.ndarray): The 3x3 rotation matrix for the current frame.
            
            t (np.ndarray): The 3x1 translation vector for the current frame.
        """
        
        _, R, t, _ = cv2.recoverPose(E, pts_2d_1, pts_2d_2, self._K)

        P = self._K @ np.hstack((R, t))

        return P, R, t
    
    def _find_matches(self, dsc1: np.ndarray, dsc2: np.ndarray) -> np.ndarray:
        """
        Finds matches between the key points of two images based on the distance of their descriptors.
        Uses Lowe's Ratio Test to filter out false positive matches.

        Args:
            dsc1 (np.ndarray): Descriptors of the first image's key points.
            dsc2 (np.ndarray): Descriptors of the second image's key points.

        Returns:
            np.ndarray: Array of matches that passed the Ratio Test, sorted by descriptor distance.
        """
        allMatches = np.array(self._matcher.knnMatch(dsc1, dsc2, k=2))
        
        good_matches = np.apply_along_axis(MiniSLAM.lowes_ratio, 1, allMatches)

        return allMatches[good_matches, 0] 
    
    def _get_matched_points(self, kp1: list, kp2: list, matches: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts feature matches into 2D point correspondences for two frames.

        Args:
            kp1 (list): List of keypoints from the first frame.
            kp2 (list): List of keypoints from the second frame.
            matches (list): List of feature matches between the keypoints of the two frames.

        Returns:
            tuple [np.ndarray, np.ndarray]: 
            - pts1 (np.ndarray): Matched 2D points from the first frame.
            - pts2 (np.ndarray): Matched 2D points from the second frame.
        """

        # Extract 2D points corresponding to the keypoints in frame 1 and frame 2 from the matches.
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)
        return pts1, pts2

    def _find_essntial(self, pts1: list, pts2: list, matches : list):
        """_summary_

        Args:
            pts1 (list): _description_
            pts2 (list): _description_
            matches (list): _description_

        Returns:
            _type_: _description_
        """
        E, mask = cv2.findEssentialMat(pts1, pts2, self._K, cv2.RANSAC, prob=0.999, threshold=0.9)
        if mask is not None:
            matches = [match for match, accepted in zip(matches, mask) if accepted]
        return E, matches
        
    def _triangulate(self, P1, P2, pts_2d_1, pts_2d_2, matches, dsc):
        if len(matches) < 5:
            print("No triangulation - Not enough matches!")
            return False
        try:
            pts_4d_homogeneous = cv2.triangulatePoints(P2, P1, pts_2d_2.T, pts_2d_1.T)
            
            pts_3d = (pts_4d_homogeneous[:3] / pts_4d_homogeneous[3]).T
            dsc_3d = np.array([dsc[m.trainIdx] for m in matches])
            
            good_pts_idxs = self._filter_points_by_distance(pts_3d, max_std=self._max_std_new_pts)
            
            pts_3d = pts_3d[good_pts_idxs]
            dsc_3d = dsc_3d[good_pts_idxs]
            
            if len(pts_3d) < 5:
                print("No triangulation - Not enough matches!")
                return False
            
            self._map3d += (pts_3d, dsc_3d)
            print(f"{len(pts_3d)} new 3d points saved")
            
        except ValueError:
            print(f"triangulation error: {ValueError}")
            return False
        return True
    
    def _filter_points_by_distance(self, points: np.ndarray, max_std: float = 2) -> np.ndarray:
        """
        Filters the points whose distance from the mean point is less than max_std standard deviations.
        
        Parameters:
        points (np.ndarray): Array of shape (n, 3) representing n 3D points.
        k (float): The number of standard deviations to use for filtering.
        
        Returns:
        np.ndarray: Array of indices of the points that meet the condition.
        """
        mean_point = np.mean(points, axis=0)
        distances = np.linalg.norm(points - mean_point, axis=1)
        std_dev = np.std(distances)
        
        return distances < (max_std * std_dev)
    
    def _essential_from_Rt(self, frame_details_prev : FrameDetails, frame_details_curr : FrameDetails):
        """
        Computes the essential matrix from frame 1 to frame 2.

        Parameters:
        R1 (np.ndarray): Rotation matrix of frame 1 (3x3).
        t1 (np.ndarray): Translation vector of frame 1 (3x1).
        R2 (np.ndarray): Rotation matrix of frame 2 (3x3).
        t2 (np.ndarray): Translation vector of frame 2 (3x1).

        Returns:
        E (np.ndarray): Essential matrix (3x3).
        """
        R1, t1 = frame_details_prev.R, frame_details_prev.t
        R2, t2 = frame_details_curr.R, frame_details_curr.t
        
        # Relative rotation and translation
        R_rel = R2 @ R1.T
        t_rel = ((R2.T @ t2) - (R1.T @ t1)).reshape(3)

        # t cross
        t_x = np.cross(np.eye(3), t_rel)

        # E = [t_rel]_x * R_rel
        return t_x @ R_rel
    
    def _get_inliers_from_essential(self, pts1, pts2, E):
        pts1 = np.hstack((pts1, np.ones((len(pts1), 1))))
        pts2 = np.hstack((pts2, np.ones((len(pts2), 1))))

        F = self._K_inv.T @ E @ self._K_inv

        return np.abs(np.sum(pts2.T * (F @ pts1.T), axis=0)) < self._triangulation_inliers_threshold
    
    def _filter_matches_new_pts(self, frame_details_curr: FrameDetails, matches_prev_curr: np.ndarray, matches_3d: np.ndarray) -> np.ndarray:
        """
        Filter out matches between the previous and current frame that correspond to unknown (new) 3D points.

        Args:
            frame_details_curr (FrameDetails): _description_
            matches_prev_curr (np.ndarray): _description_
            matches_3d (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        # Create a boolean array to track which keypoints in the current frame are not yet associated with any 3D points.
        new_kp_idxs = np.ones(len(frame_details_curr.kp), dtype=bool)

        # Mark keypoints that correspond to already-known 3D points as 'False' in the boolean array.
        for global_match in matches_3d:
            new_kp_idxs[global_match.trainIdx] = False

        # Filter out matches between the previous and current frame that correspond to unknown (new) 3D points.
        return np.array([m for m in matches_prev_curr if new_kp_idxs[m.trainIdx]])
    
    def _matches_to_prev_frame(self, frame_details_curr: FrameDetails) -> np.ndarray:
        """
        Find matches between the descriptors of the previous and current frames.
        Filter and return only the matches that are consistent with the essential matrix

        Args:
            frame_details_curr (FrameDetails): _description_

        Returns:
            np.ndarray: _description_
        """
        matches_prev_curr = self._find_matches(self._frame_details_prev.dsc, frame_details_curr.dsc)
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches_prev_curr)
        E = self._essential_from_Rt(self._frame_details_prev, frame_details_curr)
        mask = self._get_inliers_from_essential(pts_2d_prev, pts_2d_curr, E)
        return np.array([match for match, accepted in zip(matches_prev_curr, mask) if accepted])

    def remove_outliers(self, min_neighbors = 3, neighbor_dist = 0.5, min_dist = 0.005):
        self._map3d.remove_outliers(min_neighbors, neighbor_dist)
        
    def lowes_ratio(_2nn: tuple[cv2.DMatch]):
        return _2nn[0].distance < 0.77 * _2nn[1].distance
    