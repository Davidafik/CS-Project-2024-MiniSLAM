import numpy as np
from sklearn.metrics import pairwise_distances


class Map3D:
    def __init__(self, emptyDsc: np.ndarray, numpyFileName: str = None) -> None:
        # Open file and extract the data
        if numpyFileName is not None:
            f = open(numpyFileName, 'rb')
            self.pts = np.load(f)
            self.dsc = np.load(f)
        else:
            self.pts = np.empty((0, 3), np.float32)
            self.dsc = emptyDsc
            
    def load(self, numpyFileName: str):
        with open(numpyFileName, 'rb') as f:
            self.pts = np.load(f)
            self.dsc = np.load(f)
            print(f"3D map loaded successfully. \n{len(self.pts)} points found.")
        
    def save(self, numpyFileName: str):
        with open(numpyFileName, 'wb') as f:
            np.save(f, self.pts)
            np.save(f, self.dsc)
            print(f"3D map saved successfully. \n{len(self.pts)} points saved.")
            
    def __iadd__(self, morePoints: tuple): 
        """ operator += """
        self.pts = np.vstack((self.pts, morePoints[0]))
        self.dsc = np.vstack((self.dsc, morePoints[1]))
        return self 
            
    def isEmpty(self):
        return len(self.pts) == 0
    
    def remove_outliers(self, min_neighbors = 3, neighbor_dist = 0.5, min_dist = 0.005):
        if self.isEmpty():
            return
        
        dist_mat  = pairwise_distances(self.pts, self.pts, metric='euclidean', n_jobs=-1)
        pts_idxs = np.ones(len(self.pts), dtype=bool)
        
        for i, dist in enumerate(dist_mat):
            # Remove points that are too secluded.
            num_neighbors = (dist < neighbor_dist).sum() - 1
            if num_neighbors < min_neighbors:
                pts_idxs[i] = False

            # Remove points that are too close to each other.
            closest_idx = np.argpartition(dist, 1)[1]
            closest_dist = dist[closest_idx]
            if  closest_idx > i and closest_dist < min_dist:
                pts_idxs[i] = False
            
        self.pts = self.pts[pts_idxs]
        self.dsc = self.dsc[pts_idxs]
        
    def rotate_XZ(self, deg : float) -> np.ndarray:
        """
        rotate map in xz plane

        Args:
            deg (float): the rotation angle, in degrees.

        Returns:
        """
        rad = np.deg2rad(deg)
        roration_mat = np.array([[ np.cos(rad), 0, np.sin(rad)],
                                 [ 0,           1, 0          ],
                                 [-np.sin(rad), 0, np.cos(rad)]])
        self.pts = self.pts @ roration_mat
        
        
    def rotate_YZ(self, deg : float) -> np.ndarray:
        """
        rotate map in yz plane

        Args:
            deg (float): the rotation angle, in degrees.

        Returns:
        """
        rad = np.deg2rad(deg)
        roration_mat = np.array([[ 1,  0          , 0          ],
                                 [ 0,  np.cos(rad), np.sin(rad)],
                                 [ 0, -np.sin(rad), np.cos(rad)]])
        self.pts = self.pts @ roration_mat
        
