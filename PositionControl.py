from Position import Position
import numpy as np
import copy

class PositionControl:
    """
    this class is to calculate the signal we need to send to the drone,
    given the current location and the target of the drone.
    abstract class.
    """
    def __init__(self) -> None:
        """
        initialize PositionControl parameters.

        Args:
            RClimit (int, optional): a limit for RC signal.
                    makes drone motions more gentle. Defaults to 0.05.
        """
        self._lookDir = None
        self._target = None
        self._error = None


    def getRCVector(self, currPos : Position) -> tuple[np.ndarray, float]:
        pass


    def _calcErrorVec(self, currPos : Position) -> tuple[np.ndarray, float]:
        """
        get the translation and angle errors. 

        calculates a vector that point from the drone to the target.
        it describes how much the drone need to move in each direction, 
        in order to get to the target.

        also calculates how much the drone need to rotate in order to get
        to the desired position.

        Args:
            currPos (Position): the drone current position.

        Returns:
            errorVec (np.ndarray): the traslation and rotation error. (LR, DU, BF, RCW)
        """
        # the translation error.
        errorVec = self._target.getPosVec() - currPos.getPosVec()
        
        # the distance to the target.
        self._error = np.linalg.norm(errorVec[:3])

        # rotate in xz plane by -theta degrees
        errorVec[:3] = rotate_XZ(errorVec[:3], -currPos.getT())
        
        # the error in the angle of the drone
        if self._lookDir is not None:
            # make the drone look at the a certain point
            lookVec = self._lookDir.getLocVec() - currPos.getLocVec()
            desAngle = np.rad2deg(np.arctan2(lookVec[0], lookVec[2]))
            errorVec[3] = desAngle - currPos.getT()
        
        return errorVec


    def setTarget(self, target : Position) -> None:
        """
        set a new target to the drone.

        Args:
            target (Position): the new target
        """
        self._target = copy.deepcopy(target)
        self._resetError()


    def setTargetRelative(self, moveVec : Position) -> None:
        """
        set a new target relative to where the drone us now.

        Args:
            moveVec (Position): the new relative location,
                    indicates how much to move in each direction
        """
        relativeTergat = rotate_XZ(moveVec, self)
        self.setTarget(self._target + relativeTergat)


    def setLookDirection(self, lookDirection : Position = Position()) -> None:
        """
        set a point in space the the drone will try to look at while flying.

        Args:
            lookDirection (Position): the point to look at.
                defaults to [0,0,0].
        """
        self._lookDir = copy.deepcopy(lookDirection)
        self._resetError()


    def _resetError(self) -> None:
        """
        resets the error of the drone to infinity when setting a new target.
        """
        self._error = np.inf


    def getError(self) -> float:
        """
        get the distance to the target.

        Returns:
            float: the current error
        """
        return self._error



def rotate_XZ(vec : np.ndarray, deg : float) -> np.ndarray:
    """
    rotate a 3d vector in xz plane

    Args:
        vec (np.array): the vector you want to rotate
        deg (float): the rotation angle, in degrees.

    Returns:
        np.array:: the rotated vector.
    """
    rad = np.deg2rad(deg)
    rotatedVec = np.zeros(3)
    rotatedVec[0] = vec[0] * np.cos(rad) + vec[2] * np.sin(rad)
    rotatedVec[1] = vec[1]
    rotatedVec[2] = -vec[0] * np.sin(rad) + vec[2] * np.cos(rad)
    return rotatedVec
