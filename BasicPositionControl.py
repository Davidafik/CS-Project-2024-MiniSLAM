import numpy as np
from Position import Position
from PositionControl import PositionControl

class BasicPositionControl (PositionControl):
    """
    sub-class of PositionControl.
    a control with simple weights, that calibrate the 
    movment in each direction.
    """
    def __init__(
            self, 
            RClimit: np.ndarray = np.array([0.01, 0.01, 0.01, 0.1]),
            errWeights: np.ndarray = np.array([0.015, 0.02, 0.02, 0.002]),
            errMargin: np.ndarray = np.array([0.07, 0.07, 0.07, 20.0])
        ) -> None:
        """
        initialize BasicPositionControl parameters.

        Args:
            RClimit (np.ndarray, optional): a limit for RC signal. Defaults to np.array([0.05, 0.05, 0.05, 0.1]).
            
        Parameters for smoothidg the motion:
            errWeights (np.ndarray, optional): weights for tuning basic control. Defaults to np.array([0.4, 0., 0.4, 0.4]).
            errMargin (np.ndarray, optional): _description_. Defaults to np.array([1.0, 1.0, 1.0, 2.0]).
        Ordered (LR,DU,BF,RCW).
        """
        super().__init__()
        self._RClimit = RClimit
        self._errGrad = errWeights
        self._errMargin = errMargin
        self._prevPos = None

    def getRCVector(self, currPos : Position) -> tuple[np.ndarray, float]:
        """
        calculate the the rc vector to send to the drone.

        Args:
            currPos (Position): current position of the drone

        Returns:
            RCvec (np.ndarray): the rc vector to send to the drone. Ordered (LR,DU,BF,RCW).
        """
        if currPos is None:
            self._error = np.inf
            return np.zeros(4)
        
        # get the translation and angle error.
        errorVec = self._calcErrorVec(currPos) # (LR, DU, BF, RCW)

        # 
        errorVec = self.smoothError(errorVec)
        
        # limit the size RC vector
        errorVec = errorVec.clip(-self._RClimit, self._RClimit)

        if currPos == self._prevPos:
            errorVec /= 5
        self._prevPos = currPos

        return errorVec
    
    def smoothError(self, err: np.ndarray):
        """
        https://www.geogebra.org/classic/ukehexdq

        Args:
            err (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        cond = np.abs(err) < self._errMargin
        return self._errGrad * (cond * err**3 / (3 * self._errMargin**2) + np.logical_not(cond) * (err - (2 * np.sign(err) * self._errMargin / 3)))


# ctrl = BasicPositionControl(RClimit = np.array([0.00, 0.00, 0.00, 360]),
#             errWeights = np.array([0.0, 0.0, 0.0, 1]),
#             errMargin = np.array([0.001, 0.001, 0.001, 0.001]))

# ctrl.setTarget(Position([0,0,0]))
# ctrl.setLookDirection(Position([0, 0, 1]))
# print(ctrl.getRCVector(Position([1,0,0], 0)))