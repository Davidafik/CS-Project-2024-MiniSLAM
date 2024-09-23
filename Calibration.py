import numpy as np

class Calibration:
    '''
    Camera Calibration object - holds the camera clibration data
    '''

    def __init__(self, IntrinsicMatrix : np.ndarray, ExtrinsicMatrix : np.array):
        '''
        Constractor with defined matrix for the intrinsic data, and array for the extrinsic data.
        '''

        # Set the local variables
        self._camMatrix = np.copy(IntrinsicMatrix)
        self._camDist = np.copy(ExtrinsicMatrix)

        return
    
    def __init__(self, numpyFileName):
        '''
        Constractor to load calibration data from numpy file.
        '''

        # Open file and extract the data
        with open(numpyFileName, 'rb') as f:
            self._camMatrix = np.load(f)
            self._camDist = np.load(f)
        
        return
    
    
    def getIntrinsicMatrix(self):
        '''
        Get intrinsic matrix of the camera calibration
        '''
        return np.copy(self._camMatrix)
    

    def getExtrinsicMatrix(self):
        '''
        Get extrinsic matrix of the camera calibration
        '''
        return np.copy(self._camDist)