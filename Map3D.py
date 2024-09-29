import numpy as np

class Map3D:
    def __init__(self, numpyFileName: str = None) -> None:
        # Open file and extract the data
        if numpyFileName is not None:
            f = open(numpyFileName, 'rb')
            self.pts = np.load(f)
            self.dsc = np.load(f)
        else:
            self.pts = np.empty((0, 3), np.float32)
            self.dsc = np.empty((0, 128), np.float32)
            
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
        return len(self.pts) is 0
