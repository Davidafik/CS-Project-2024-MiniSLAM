import numpy as np
import cv2
from sklearn.metrics import pairwise_distances
from FrameDetails import FrameDetails
from Map3D import Map3D

class Mapping:
    def __init__(self, K, distCoeffs = None, map_3d_path = None, add_new_pts: bool = True, max_std_new_pts: float = 2) -> None:
        self._K = K
        self._K_inv = np.linalg.inv(K)
        self._distCoeffs = distCoeffs
        
        self._feature_detector = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher()
        
        self._map3d = Map3D(map_3d_path)
        self._frame_details_prev = None
        
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
        # Convert the current RGB frame to grayscale, as the SIFT detector operates on single-channel images.
        gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

        # Detect keypoints and compute descriptors using SIFT on the grayscale image.
        kp, dsc = self._feature_detector.detectAndCompute(gray_new_frame, None)

        # Create a new FrameDetails object to store the current frame's keypoints, descriptors, and pose.
        frame_details_curr = FrameDetails(key_points=kp, descriptors=dsc)
        
        # If we have already have 3d points cloud, attempt localization with PnP, and triangulate new 3d points.
        if not self._map3d.isEmpty():
            
            ##### Part 1: Attempt to localize the current frame using PnP (Perspective-n-Point) with the global 3D map. #####
            
            matches_3d = self._localize_with_PnP(frame_details_curr)
            
            # If localization failed, return None.
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
            if dist_to_prev < 1 or dist_to_prev > 15:
                print(f"no triangulation - camera movement is {dist_to_prev}")
                return frame_details_curr
            
            # Find feature matches between the previous and current frames
            matches_prev_curr = self._matches_to_prev_frame(frame_details_curr)

            # Filter matches between the previous and current frame that correspond to unknown (new) 3D points
            new_matches_prev_curr = self._filter_matches_new_pts(frame_details_curr, matches_prev_curr, matches_3d)

            # Get the lists of the 2D points from the matches.
            pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, new_matches_prev_curr)
                        
            # Triangulate new 3D points from the 2D correspondences and add them to the global 3D map.
            self._triangulate(self._frame_details_prev.P, frame_details_curr.P, pts_2d_prev, pts_2d_curr, new_matches_prev_curr, dsc)

            # Store the frame for matching with the next one.
            self._frame_details_prev = frame_details_curr

            # Store the matches for potential use elsewhere (e.g., for visualization or debugging).
            self._matches = new_matches_prev_curr
            
            # Return the current frame's details after processing.
            return frame_details_curr
        
        # If this is the second frame being processed, attempt to localize using feature matches between the first and the current frames.
        elif self._frame_details_prev is not None:
            
            ##### Part 1: Attempt to localize the current frame using feature matches between the first current frame. #####
            
            # Attempt to localize using feature matches between the first frame and the current frame.
            pts_2d_first, pts_2d_curr, matches_prev_curr = self._localize_with_prev_frame(frame_details_curr)
            
            # If localization failed, return None.
            if matches_prev_curr is None:
                return None
            
            ##### Part 2: Triangulate 3d points based on the matched features found in part 1: #####
            
            if not self.add_new_pts:
                return frame_details_curr
            
            # Triangulate new 3D points between the first and current frames.
            ret = self._triangulate(self._frame_details_prev.P, frame_details_curr.P, pts_2d_first, pts_2d_curr, matches_prev_curr, dsc)
            
            # If triangulation fails, return None.
            if not ret:
                return None
            
            # Store the frame for matching with the next one.
            self._frame_details_prev = frame_details_curr
            
            # Store the matches for potential use elsewhere (e.g., for visualization or debugging).
            self._matches = matches_prev_curr
            
            # Return the current frame's details.
            return frame_details_curr

        # If this is the very first frame being processed, initialize its pose as the identity.
        else:
            # Set the rotation matrix to the identity matrix (no rotation).
            frame_details_curr.R = np.eye(3)

            # Set the translation vector to zero (camera located at the origin).
            frame_details_curr.t = np.zeros((3, 1))

            # Compute the initial projection matrix using the intrinsic camera matrix.
            frame_details_curr.P = np.hstack((self._K, np.zeros((3, 1))))

            # Store the frame for matching with the next one.
            self._frame_details_prev = frame_details_curr

            # Return the current frame's details.
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
        # Find feature matches between the global 3D descriptors and the current frame's descriptors.
        matches_3d_curr = self._find_matches(self._map3d.dsc, frame_details_curr.dsc)

        # Check if there are enough matches to attempt PnP. At least 5 matches are required for solving PnP.
        if len(matches_3d_curr) < 5:
            print("Not enough matches!")
            return None
        
        # Extract the corresponding 3D points from the global map based on the matched 3D descriptors.
        pts_3d = np.array([self._map3d.pts[m.queryIdx] for m in matches_3d_curr])

        # Extract the corresponding 2D points from the current frame's keypoints based on the matched descriptors.
        pts_2d_curr = np.array([frame_details_curr.kp[m.trainIdx].pt for m in matches_3d_curr], dtype=np.float64)

        try:
            # Solve the PnP problem using RANSAC to estimate the rotation (rvec) and translation (t) of the camera.
            success, rvec, t, _ = cv2.solvePnPRansac(
                objectPoints=pts_3d,              # 3D points in the global map
                imagePoints=pts_2d_curr,          # Corresponding 2D points in the current frame
                cameraMatrix=self._K,             # Intrinsic camera matrix
                distCoeffs=self._distCoeffs,      # Distortion coefficients of the camera
                flags=cv2.SOLVEPNP_ITERATIVE,     # Method for solving PnP
                confidence=0.8,                 # Confidence level for RANSAC
                # reprojectionError=0.8,            # Maximum allowed reprojection error
                # rvec=cv2.Rodrigues(self._frame_details_prev.R)[0],
                # tvec=self._frame_details_prev.t,
                # useExtrinsicGuess=True
            )
        
        except ValueError:
            # Handle potential errors during the PnP solving process.
            print(f"PnP error! {ValueError}")
            return None

        # If PnP fails (e.g., if RANSAC cannot find a solution), return None.
        if not success:
            print("PnP failed!")
            return None

        # Convert the rotation vector (rvec) into a rotation matrix.
        R, _ = cv2.Rodrigues(rvec)

        # Compute the projection matrix P using the intrinsic camera matrix and the rotation and translation.
        P = self._K @ np.hstack((R, t))

        # Update the current frame's rotation, translation, and projection matrix with the computed values.
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
        # Find matches between the descriptors of the previous and current frames.
        matches = self._find_matches(self._frame_details_prev.dsc, frame_details_curr.dsc)

        # Check if there are enough matches (at least 5) to proceed.
        if len(matches) < 5:
            print("Not enough matches!")
            return None, None, None

        # Get the matched 2D points from the keypoints of the previous and current frames.
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches)

        # Compute the essential matrix and refine the matches based on the epipolar constraint.
        E, matches = self._find_essntial(pts_2d_prev, pts_2d_curr, matches)

        # Recompute the matched points based on the refined matches (filtered by the essential matrix).
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches)

        # Estimate the current frame's pose (rotation and translation) using the essential matrix.
        P, R, t = self._calcPosition(E, pts_2d_prev, pts_2d_curr, False)

        # Update the current frame's rotation, translation, and projection matrix with the computed values.
        frame_details_curr.R = R
        frame_details_curr.t = t
        frame_details_curr.P = P

        # Return the matched 2D points from the previous and current frames, as well as the refined matches.
        return pts_2d_prev, pts_2d_curr, matches

    def _calcPosition(self, E: np.ndarray, pts_2d_1: np.ndarray, pts_2d_2: np.ndarray, normalization: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
        # Recover the rotation (R) and translation (t) from the essential matrix using the 2D points.
        _, R, t, _ = cv2.recoverPose(E, pts_2d_1, pts_2d_2, self._K)

        # Optionally normalize the translation vector to have unit length.
        if normalization:
            t = t / np.linalg.norm(t)

        # Construct the projection matrix P by combining the intrinsic matrix K with R and t.
        P = self._K @ np.hstack((R, t))

        # Return the projection matrix (P), rotation matrix (R), and translation vector (t).
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

        # Find the two best matches for each descriptor using KNN matching.
        allMatches = self._matcher.knnMatch(dsc1, dsc2, k=2)

        matches = []

        # Apply Lowe's Ratio Test to retain only good matches.
        for m, n in allMatches:
            if m.distance < 0.8 * n.distance:
                matches.append(m)

        # # Sort the matches by their distance to prioritize the closest matches.
        # matches = sorted(matches, key=lambda x: x.distance)

        # Return the matches as a NumPy array.
        return np.array(matches)
    
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

        # Return the matched 2D points from both frames as NumPy arrays.
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
        # Only triangulate new points if there are enough valid matches (at least 5).
        if len(matches) < 5:
            return False
            
        try:
            # Triangulate new 3D points using the projection matrices of the first and current frames
            pts_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts_2d_1.T, pts_2d_2.T)
            
            pts_3d = (pts_4d_homogeneous[:3] / pts_4d_homogeneous[3]).T
            dsc_3d = np.array([dsc[m.trainIdx] for m in matches])
            
            good_pts_idxs = self._filter_points_by_distance(pts_3d, max_std=self._max_std_new_pts)
            
            pts_3d = pts_3d[good_pts_idxs]
            dsc_3d = dsc_3d[good_pts_idxs]
            
            if len(pts_3d) < 5:
                return False
            print(f"{len(pts_3d)} new 3d points saved")
            
            self._map3d += (pts_3d, dsc_3d)
            
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
        # Compute the relative rotation
        R_rel = R2 @ R1.T

        # Compute the relative translation
        t_rel = (t2 - R2 @ R1.T @ t1).reshape(3)

        # Compute the skew-symmetric matrix [t_rel]_x
        t_x = np.array([[0, -t_rel[2], t_rel[1]],
                        [t_rel[2], 0, -t_rel[0]],
                        [-t_rel[1], t_rel[0], 0]])

        # Compute the essential matrix: E = [t_rel]_x * R_rel
        E = t_x @ R_rel

        return E
    
    def _get_inliers_from_essential(self, pts1, pts2, E, threshold=1e-2):
        F = self._K_inv.T @ E @ self._K_inv
        
        pts1 = np.hstack((pts1, np.ones((len(pts1), 1))))
        pts2 = np.hstack((pts2, np.ones((len(pts2), 1))))

        epipolar_constraint = np.abs(np.sum(pts2.T * (F @ pts1.T), axis=0))
        
        return epipolar_constraint < threshold
    
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
        # Find matches between the descriptors of the previous and current frames.
        matches_prev_curr = self._find_matches(self._frame_details_prev.dsc, frame_details_curr.dsc)

        # Extract the 2D points corresponding to the matched keypoints in both the previous and current frames.
        pts_2d_prev, pts_2d_curr = self._get_matched_points(self._frame_details_prev.kp, frame_details_curr.kp, matches_prev_curr)

        # Compute the essential matrix between the two frames (based on their rotation and translation that we found with PnP).
        E = self._essential_from_Rt(self._frame_details_prev, frame_details_curr)

        # Filter the matches that are consistent with the essential matrix (i.e., satisfy the epipolar constraint).
        mask = self._get_inliers_from_essential(pts_2d_prev, pts_2d_curr, E)

        # Retain only the matches that passed the epipolar constraint test.
        return np.array([match for match, accepted in zip(matches_prev_curr, mask) if accepted])

    def remove_outliers(self, min_neighbors = 3, neighbor_dist = 0.5, min_threshold = 1e-1):
        dist_mat  = pairwise_distances(self._map3d.pts, self._map3d.pts, metric='euclidean', n_jobs=1)
        pts_idxs = np.ones(len(self._map3d.pts), dtype=bool)
        
        for i, dist in enumerate(dist_mat):
            closest_idx = np.argpartition(dist, 1)[1]
            num_neighbors = (dist < neighbor_dist).sum()
            closest_dist = dist[closest_idx]
            if num_neighbors < min_neighbors:
                pts_idxs[i] = False
            if  closest_idx > i and closest_dist < min_threshold:
                pts_idxs[i] = False
            
        self._map3d.pts = self._map3d.pts[pts_idxs]
        self._map3d.dsc = self._map3d.dsc[pts_idxs]
        print(len(self._map3d.pts))
                    