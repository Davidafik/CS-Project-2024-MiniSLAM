import numpy as np
from collections import defaultdict

class FrameDetails:
    def __init__(self, key_points : list, descriptors : list, R : np.ndarray = None, t : np.ndarray = None, P : np.ndarray = None):
        self.key_points = key_points
        self.descriptors = descriptors
        self.R = R
        self.t = t
        self.P = P
