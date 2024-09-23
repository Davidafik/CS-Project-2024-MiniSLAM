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
        frame_details_curr = FrameDetails(key_points=kp, descriptors=des)
        
        if len(self._all_frames) > 1:
            P, R, t = self._localize_with_PnP(frame_details_curr)
            
            if P is None:
                return None
            print(f"R PnP: \n{R}")
            
            frame_details_curr.R = R
            frame_details_curr.t = t
            frame_details_curr.P = P
            
            self._all_frames.append(frame_details_curr)
            
            frame_details_prev = self._all_frames[-1]
            pts_2d_first, pts_2d_curr, matches_prev_curr, P, R, t = self._localize_with_prev_frame(frame_details_prev, frame_details_curr)
            if matches_prev_curr is None: 
                return frame_details_curr
            print(f"R prev: \n{R}")
            # ret = self._triangulate(frame_details_prev.P, frame_details_curr.P, pts_2d_first, pts_2d_curr, matches_prev_curr, des)
            return frame_details_curr
            
        elif len(self._all_frames) == 1:
            # Find feature matches between first and current frames
            frame_details_prev = self._all_frames[-1]
            pts_2d_first, pts_2d_curr, matches_prev_curr, P, R, t = self._localize_with_prev_frame(frame_details_prev, frame_details_curr)
            
            if matches_prev_curr is None: 
                return None
            
            frame_details_curr.R = R
            frame_details_curr.t = t
            frame_details_curr.P = P
            
            ret = self._triangulate(frame_details_prev.P, frame_details_curr.P, pts_2d_first, pts_2d_curr, matches_prev_curr, des)
            if ret:
                self._all_frames.append(frame_details_curr)
            self._matches = matches_prev_curr
            return frame_details_curr

        else:
            frame_details_curr.R = np.eye(3)
            frame_details_curr.t = np.zeros((3,1))
            frame_details_curr.P = np.hstack((self._K, np.zeros((3,1))))
            self._all_frames.append(frame_details_curr)
            return frame_details_curr
        
    def _localize_with_PnP(self, frame_details_curr : FrameDetails):
        """
        Solve PnP to get rotation and translation

        Args:
            pts_3d (_type_): _description_
            frame_details_curr (_type_): _description_
            pts_2d_curr (_type_): _description_

        Returns:
            _type_: _description_
        """
        matches_3d_curr = self._find_matches(self._global_3d_des, frame_details_curr.descriptors)
        
        if len(matches_3d_curr) < 5:
            print("not enough matches!")
            return None, None, None
        
        pts_3d = np.array([self._global_3d_pts[m.queryIdx] for m in matches_3d_curr])
        pts_2d_curr = np.array([frame_details_curr.key_points[m.trainIdx].pt for m in matches_3d_curr], dtype=np.float64)
            
        try:
            success, rvec, t, _ = cv2.solvePnPRansac(
                objectPoints=pts_3d, 
                imagePoints=pts_2d_curr, 
                cameraMatrix=self._K, 
                distCoeffs=self._distCoeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except:
            print("PnP error!")
            return None, None, None
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            P = self._K @ np.hstack((R, t))
            
            return P, R, t
        else:
            print("PnP failed!")
            return None, None, None
        
    def _localize_with_prev_frame(self, frame_details_prev : FrameDetails, frame_details_curr : FrameDetails):
        matches = self._find_matches(frame_details_prev.descriptors, frame_details_curr.descriptors)
        if len(matches) < 5:
            print("not enough matches!")
            return None, None, None, None, None, None
        pts_2d_prev, pts_2d_curr = self._get_matched_points(frame_details_prev.key_points, frame_details_curr.key_points, matches)
        E, matches = self._find_essntial(pts_2d_prev, pts_2d_curr, matches)
        pts_2d_prev, pts_2d_curr = self._get_matched_points(frame_details_prev.key_points, frame_details_curr.key_points, matches)
        P, R, t = self._calcPosition(frame_details_curr, E, pts_2d_prev, pts_2d_curr)
        return pts_2d_prev, pts_2d_curr, matches, P, R, t
        
    def _calcPosition(self, frame_details : FrameDetails, E, pts_2d_1, pts_2d_2, normalization : bool = True):
        _, R, t, _ = cv2.recoverPose(E, pts_2d_1, pts_2d_2, self._K)
        if normalization:
            t = t / np.linalg.norm(t)
        P = self._K @ np.hstack((R, t))
        return P, R, t
        
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
    
    