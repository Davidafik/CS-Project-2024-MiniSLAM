import cv2 as cv
import threading

class VideoCapture:
    '''
    If your video capture is in delay for some reason, and you only care about the last frame,
    (and dont care about loosing some frames), this cv.VideoCapture is for you!
    The VideoCapture work the same as Video capture,
    but with the exaption the it allways give the last frame!
    '''
    def __init__ (self, source):
        '''
        Initate Video capturing as in VideoCapture
        '''
            # Set Camera object (with openCV VideoCapture), and get te first frame
        self._camera = cv.VideoCapture(source)
        self._ret, self._frame = self._camera.read()

        self._live = True

        # Start thread to run in the background and retrive frames
        self._thread = threading.Thread(target=self.__ReadFrames__, args=(), name="rtsp_read_thread")
        self._thread.daemon = True
        self._thread.start()
    
    def __ReadFrames__(self):
        '''
        Private method to work in the background and reading the newest frame
        '''
        while self._live:
            self._ret, self._frame = self._camera.read()

    def read(self):
        '''
        As in VideoCapture, read the last frame
        '''
        return self._ret, self._frame
    
    def release(self):
        '''
        Stop the Video streaming
        '''
        self._live = False
        self._thread.join()