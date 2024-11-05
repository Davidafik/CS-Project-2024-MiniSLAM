import cv2
import numpy as np
        
class FeatureDetector:
    def detectAndCompute(self, image: np.ndarray):
        pass
    
    def emptyDsc(self) -> np.ndarray:
        pass


class SIFT (FeatureDetector):
    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6) -> None:
        super().__init__()
        self._feature_detector = cv2.SIFT_create(nfeatures=nfeatures, 
                                                 nOctaveLayers=nOctaveLayers, 
                                                 contrastThreshold=contrastThreshold, 
                                                 edgeThreshold=edgeThreshold, 
                                                 sigma=sigma
                                                )
        
    def detectAndCompute(self, image: np.ndarray):                                        
        return self._feature_detector.detectAndCompute(image, None)
        
    def emptyDsc(self) -> np.ndarray:
        return np.empty((0, 128), np.float32)


class ORB (FeatureDetector):
    def __init__(self, nfeatures=5000, scaleFactor=1.2, nlevels=10, edgeThreshold=30, firstLevel=0, WTA_K=3, scoreType=cv2.ORB_FAST_SCORE, patchSize=42, fastThreshold=20) -> None:
        super().__init__()
        self._feature_detector = cv2.ORB_create(nfeatures=nfeatures,            # Maximum number of features to retain
                                                scaleFactor=scaleFactor,        # Pyramid decimation ratio
                                                nlevels=nlevels,                # Number of pyramid levels
                                                edgeThreshold=edgeThreshold,    # Minimum distance from edge for keypoints
                                                firstLevel=firstLevel,          # Start level of the pyramid
                                                WTA_K=WTA_K,                    # Number of points that produce each element of the descriptor
                                                scoreType=scoreType,            # Score to rank the keypoints
                                                patchSize=patchSize,            # Size of the patch used by ORB
                                                fastThreshold=fastThreshold     # Threshold for FAST keypoint detector
                                               )
    
    def detectAndCompute(self, image: np.ndarray):                                        
        return self._feature_detector.detectAndCompute(image, None)
        
    def emptyDsc(self) -> np.ndarray:
        return np.empty((0, 32), np.uint8)


class AKAZE (FeatureDetector):
    def __init__(self, descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=256, descriptor_channels=3, threshold=0.0001, nOctaves=4, nOctaveLayers=4) -> None:
        super().__init__()
        self._feature_detector = cv2.AKAZE_create(descriptor_type=descriptor_type,  # Type of descriptor (MLDB is binary and faster)
                                                  descriptor_size=descriptor_size,  # Size of descriptor in bits, 0 uses full size
                                                  descriptor_channels=descriptor_channels,   # Number of channels (1, 2, or 3)
                                                  threshold=threshold,              # Detector response threshold to select keypoints
                                                  nOctaves=nOctaves,                # Number of octaves for detection
                                                  nOctaveLayers=nOctaveLayers       # Number of layers within each octave
                                                 )
        self._dscSize = descriptor_size // 8
    
    def detectAndCompute(self, image: np.ndarray):                                        
        return self._feature_detector.detectAndCompute(image, None)
        
    def emptyDsc(self) -> np.ndarray:
        return np.empty((0, self._dscSize), np.uint8)


class FAST_SIFT(FeatureDetector):
    def __init__(self, threshold=25, nonmaxSuppression=True, max_features=3000, nOctaveLayers=3, sigma=1.6) -> None:
        super().__init__()
        self._fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSuppression)
        self._sift = cv2.SIFT_create(
            nOctaveLayers=nOctaveLayers,
            sigma=sigma
        )
        self._max_features = max_features

    def detectAndCompute(self, image: np.ndarray):
        
        keypoints = self._fast.detect(image, None)

        # Limit the number of FAST keypoints to `self._max_features`
        if self._max_features and len(keypoints) > self._max_features:
            # Create a structured array with (response, index)
            responses = np.array([(kp.response, i) for i, kp in enumerate(keypoints)], dtype=[('response', np.float32), ('index', np.int32)])
            # Use np.partition to select the top responses
            top_responses = np.partition(responses, -self._max_features)[-self._max_features:]
            # Retrieve the top keypoints based on indices
            keypoints = [keypoints[i] for i in top_responses['index']]

        return self._sift.compute(image, keypoints)

    def emptyDsc(self) -> np.ndarray:
        return np.empty((0, 128), np.float32)


