import cv2

class FaceDetectionBaseline(object):
    def __init__(self):
        self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def input_transform(self, input_image):
        pass

    def detect_face(self, input_image):
        """
        input: Image object
        output: list of coordinates (x0, y0, width, height )
        """
        # makes pict
        gray = cv2.cvtColor(input_image.img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.classifier.detectMultiScale(gray, 1.1, 4)
        # returns
        return faces

class BetterFaceDetection(FaceDetectionBaseline):
    def __init__(self):
        super().__init__()
