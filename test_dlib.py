import dlib 
import cv2

detector = dlib.get_frontal_face_detectior()
webcam  = cv2.VideoCapture(0)