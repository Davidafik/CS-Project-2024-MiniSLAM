import numpy as np
import cv2

from FrameDetails import FrameDetails

class Mapping:
    def __init__(self, K, distCoeffs = None) -> None:
        self._K = K
        self._distCoeffs = distCoeffs
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher()
        
        self._global_3d_pts = np.empty((0, 3), np.float32)
        self._global_3d_des = np.empty((0, 128), np.float32)
        
        self._all_frames = []
        
    def process_frame(self, new_frame) -> FrameDetails:
        gray_new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        kp, des = self._sift.detectAndCompute(gray_new_frame, None)
        new_frame_details = FrameDetails(key_points=kp, descriptors=des)
        
        
        if len(self._all_frames) > 1:
            matches = self._find_matches(self._global_3d_des, new_frame_details.descriptors)
            if len(matches) < 5:
                print("not enough matches!")
                return None
            
            pts_3d = np.array([self._global_3d_pts[m.queryIdx] for m in matches])
            pts_2d_curr = np.array([new_frame_details.key_points[m.trainIdx].pt for m in matches], dtype=np.float64)
            
            ret = self._localize_PnP(new_frame_details, pts_3d, pts_2d_curr)
            if not ret:
                return None
            
            self._all_frames.append(new_frame_details)
            self._matches = matches
            
            
            return new_frame_details
            
        
        elif len(self._all_frames) == 1:
            # Find feature matches between first and current frames
            first_frame_details = self._all_frames[-1]
            matches = self._find_matches(first_frame_details.descriptors, new_frame_details.descriptors)
            if len(matches) < 5:
                print("not enough matches!")
                return None
            
            pts_2d_first, pts_2d_curr = self._get_matched_points(first_frame_details.key_points, new_frame_details.key_points, matches)

            # Estimate the essential matrix using RANSAC and recover pose
            E, matches = self._find_essntial(pts_2d_first, pts_2d_curr, matches)
            
            pts_2d_first, pts_2d_curr = self._get_matched_points(first_frame_details.key_points, new_frame_details.key_points, matches)

            self._calcPosition(new_frame_details, E, pts_2d_first, pts_2d_curr)
            
            ret = self._triangulate(first_frame_details.P, new_frame_details.P, pts_2d_first, pts_2d_curr, matches, des)
            
            if ret:
                self._all_frames.append(new_frame_details)
            self._matches = matches
            
            return new_frame_details

        else:
            new_frame_details.R = np.eye(3)
            new_frame_details.t = np.zeros((3,1))
            new_frame_details.P = np.hstack((self._K, np.zeros((3,1))))
            self._all_frames.append(new_frame_details)
            return new_frame_details
        
    def _triangulate(self, P1, P2, pts_2d_1, pts_2d_2, matches, des):
        try:
            # Triangulate new 3D points using the projection matrices of the first and current frames
            pts_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts_2d_1.T, pts_2d_2.T)
            
            pts_3d = pts_4d_homogeneous[:3] / pts_4d_homogeneous[3]
            des_3d = np.array([des[m.trainIdx] for m in matches])
            
            self._global_3d_pts = np.vstack((self._global_3d_pts, pts_3d.T))
            self._global_3d_des = np.vstack((self._global_3d_des, des_3d))
            
            # print(f"num 3d pts: {self._global_3d_pts.shape}")
            # print(f"num 3d des: {self._global_3d_des.shape}")
            # print(f"3d des: \n{self._global_3d_des}")
            
        except:
            print("error")
            return False
        return True
        
    def _calcPosition(self, frame_details : FrameDetails, E, pts_2d_1, pts_2d_2, normalization : bool = True):
        _, R, t, _ = cv2.recoverPose(E, pts_2d_1, pts_2d_2, self._K)
        if normalization:
            t = t / np.linalg.norm(t)

        frame_details.R = R
        frame_details.t = t
        frame_details.P = self._K @ np.hstack((R, t))
        
    def _localize_PnP(self, new_frame_details, pts_3d, pts_2d_curr):
        try:
            # Solve PnP to get rotation and translation
            success, rvec, t, _ = cv2.solvePnPRansac(
                objectPoints=pts_3d, 
                imagePoints=pts_2d_curr, 
                cameraMatrix=self._K, 
                distCoeffs=self._distCoeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except:
            print("PnP error!")
            return False
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            new_frame_details.R = R
            new_frame_details.t = t
            new_frame_details.P = self._K @ np.hstack((R, t))
            
            return True
        else:
            print("PnP failed!")
            return False
        
    def _find_matches(self, dsc1 : np.ndarray, dsc2 : np.ndarray) -> np.ndarray:
        """
        Finds matches between the Key points of the two images, based on the distance of their descriptors.
        Uses the Ratio Test to reduce False Positive results.

        Args:
            dsc1 (np.ndarray): Descriptors of the first image's key points.
            dsc2 (np.ndarray): Descriptors of the second image's key points.

        Returns:
            np.ndarray: The matches found.
        """

        allMatches = self._matcher.knnMatch(dsc1, dsc2, k=2)

        matches = []
        for m, n in allMatches:
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        matches = sorted(matches, key = lambda x:x.distance)
        return np.array(matches)
    
    def _get_matched_points(self, kp1: list, kp2: list, matches: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts feature matches into 2D point correspondences for two frames.

        Args:
            kp1 (list): Keypoints from frame 1.
            kp2 (list): Keypoints from frame 2.
            matches (list): List of feature matches.

        Returns:
            tuple: Matched 2D points from the two frames.
        """
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)
        return pts1, pts2
    
    def _find_essntial(self, pts1: list, pts2: list, matches : list):
        E, mask = cv2.findEssentialMat(pts1, pts2, self._K, cv2.RANSAC, prob=0.999, threshold=0.9)
        if mask is not None:
            matches = [match for match, accepted in zip(matches, mask) if accepted]
        return E, matches
    
    