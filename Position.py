import numpy as np

class Position:
    """
    a data structure to hold a position (x,y,z) in space, 
    and an angle (theta) on the XZ plane clock-wise.
    """
    def __init__(self, position : np.ndarray = np.array([0.0, 0.0, 0.0]), theta : float = 0.0) -> None:
        """
        defines new Position

        Args:
            position (np.ndarray, optional): [x, y, z]. 
                    Defaults to np.array([0.0, 0.0, 0.0]).
            theta (float, optional): angle in deg, on the XZ plane clock-wise (when the X axis is zero).
                    Defaults to 0.0.
        """
        self._x = position[0]
        self._y = position[1]
        self._z = position[2]
        self._t = theta
    
    def getPosVec(self) -> np.ndarray:
        """ Get 4D vector with the position data [x, y, z, theta] """
        return np.array([self._x, self._y, self._z, self._t])
     
    def getLocVec(self) -> np.ndarray:
        """ Get 3D vector with the location data [x, y, z] """
        return np.array([self._x, self._y, self._z])
    
    # Get the X coordinate (in cm)
    def getX(self) -> float:
        return self._x
    
    # Get the Y coordinate (in cm)
    def getY(self) -> float:
        return self._y
    
    # Get the Z coordinate (in cm)
    def getZ(self) -> float:
        return self._z
    
    # Get the theta angle (in deg)
    def getT(self) -> float:
        return self._t
    
    # Set the X coordinate (in cm)
    def setX(self, x : float) -> None:
        self._x = x
    
    # Set the Y coordinate (in cm)
    def setY(self, y : float) -> None:
        self._y = y
    
    # Set the Z coordinate (in cm)
    def setZ(self, z : float) -> None:
        self._z = z
    
    # Set the theta angle (in deg)
    def setT(self, t : float) -> None:
        self._t = t % 180
    
    def __repr__(self) -> str:
        return f"x:{self._x}\t y:{self._y}\t z:{self._z}\t t:{self._t}"
    
    def __add__(self, other):
        x = self._x + other.getX()
        y = self._y + other.getY()
        z = self._z + other.getZ()
        t = self._t + other.getT()
        return Position([x, y, z], t)
    
    def __sub__(self, other):
        x = self._x - other.getX()
        y = self._y - other.getY()
        z = self._z - other.getZ()
        t = self._t - other.getT()
        return Position([x, y, z], t)
    
    def __mul__(self, scalar):
        x = self._x * scalar
        y = self._y * scalar
        z = self._z * scalar
        t = self._t * scalar
        return Position([x, y, z], t)
    
    def __truediv__(self, scalar):
        x = self._x / scalar
        y = self._y / scalar
        z = self._z / scalar
        t = self._t / scalar
        return Position([x, y, z], t)