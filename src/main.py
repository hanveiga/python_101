import cv2
import sys
from models import FaceDetectionBaseline
from data import Image

def main(file_path):
    #load image
    input_image = Image(file_path)

    #initialise classifier
    classifier = FaceDetectionBaseline()

    # perform classification
    coordinates = classifier.detect_face(input_image)

    # plot result
    for (x, y, w, h) in coordinates:
        cv2.rectangle(input_image.img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    input_image.show()
    cv2.waitKey()

if __name__=='__main__':
    file_path = sys.argv[1]
    main(file_path)
