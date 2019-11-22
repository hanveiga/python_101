import cv2

class Image(object):
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        # additional features which might be relevant to keep?

    def show(self):
        cv2.imshow('img',self.img)

    # additional operations on your image objects?
