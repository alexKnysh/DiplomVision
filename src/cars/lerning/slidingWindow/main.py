from cv2 import *
from slidingWindow.slidingWindow import SlidingWindow

if __name__ == '__main__':
    print '-----start-------'
    img = imread('/home/aknysh/mbuntu (16).jpg')
    x = SlidingWindow()
    x.sliding_windows(img)
    imshow('test', img)
    waitKey()
    print '------end--------'
