import numpy as np
import yaml

path = 'Camera Calibration/Images5'



############ Example load data from '.yaml' ############

# Load the data from the 'yaml' file
with open(path + '/Calibration.yaml', 'r') as f:
    # Load the data
    mtx, dist = yaml.load(f, Loader=yaml.FullLoader)

    # Convert the data (as needed)
    mtx = np.matrix(mtx)
    dist = np.array(dist)

    # Print the data
    print (mtx)
    print (dist)



############ Example load data from '.npy' ############

# Load the data from the 'npy' file
with open(path + '/Calibration.npy', 'rb') as f:
    # Load the data
    mtx = np.load(f)
    dist = np.load(f)

    # Print the data
    print (mtx)
    print (dist)