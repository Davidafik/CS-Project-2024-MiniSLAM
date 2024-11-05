from OpenDJI import OpenDJI

"""
In this example you will see how to use the 'help' functions.
The help function help the user to find and understand the available
commands that the MSDK offer.
"""

# IP address of the connected android device
IP_ADDR = "10.0.0.3"


# Connect to the drone
with OpenDJI(IP_ADDR) as drone:
    
    # Get list of available modules
    list_modules = drone.getModules()[1:-1].replace('"', '').split(",")
    print("Modules :", list_modules)
    print()

    # Get list of available keys inside a module
    list_keys = drone.getModuleKeys(OpenDJI.MODULE_GIMBAL)[1:-1].replace('"', '').split(",")
    print("Module Keys :", sorted(list_keys))
    print()

    # Get information on specific key
    key_info = drone.getKeyInfo(OpenDJI.MODULE_GIMBAL, "YawRelativeToAircraftHeading")
    print("Key Info :")
    print(key_info)

    # command_argument = ('{'
    #             '"mode":65535,'
    #             f'"pitch":{-90:5},'
    #             f'"roll":{0:5},'
    #             f'"yaw":{0:5},'
    #             '"pitchIgnored":false,'
    #             '"rollIgnored":false,'
    #             '"yawIgnored":false,'
    #             '"duration":0,'
    #             '"jointReferenceUsed":false,'
    #             '"timeout":10'
    #         '}')
            
    # print(drone.action(OpenDJI.MODULE_GIMBAL, "RotateByAngle", command_argument))
    print()