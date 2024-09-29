import numpy as np
from collections import defaultdict

class FrameDetails:
    def __init__(self, key_points : list, descriptors : list, R : np.ndarray = None, t : np.ndarray = None, P : np.ndarray = None):
        self.kp = key_points
        self.dsc = descriptors
        self.R = R
        self.t = t
        self.P = P
