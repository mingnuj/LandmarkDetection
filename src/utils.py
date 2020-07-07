import numpy as np
import cv2
import dlib
import sys
import face_alignment

class FaceDetector(object):
    def __init__(self, detector = None, cascade_file = "../lbpcascade_animeface.xml"):
        super(FaceDetector, self).__init__()
        if detector is None:
            self.detector = dlib.get_frontal_face_detector()
            dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.shape_predictor(detector)
        self.cascade = cv2.CascadeClassifier(cascade_file)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 3 means upsampled 2 times. It makes everything bigger, and allows to detect more
        return self.detector(gray, 0)

    def detect_anim(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detected = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        faces = []

        for face in detected:
            left = face[0] - 50
            top = face[1] - 50
            right = face[0] + face[2] + 50
            bottom = face[1] + face[3] + 50
            faces.append(dlib.rectangle(left, top, right, bottom))
        return faces

class LandmarkExtractor(object):
    def __init__(self, landmark_model_file = None, dim = "2D"):
        super(LandmarkExtractor, self).__init__()

        if (landmark_model_file == None) & (dim == "2D"):
            self.FANET = True
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
        elif (landmark_model_file == None) & (dim == "3D"):
            self.FANET = True
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda')
        else:
            print("with model file")
            self.FANET = False
            self.fa = dlib.shape_predictor(landmark_model_file)

    def reshape_for_polyline(self, array):
        if np.asarray(array).shape[1] == 2:
            return np.array(array, np.int32).reshape((-1, 1, 2))
        else:
            return np.array(array, np.int32).reshape((-1, 1, 3))

    def landmark_extractor(self, image, faces):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (len(faces) is not 0) & self.FANET:
            face_boxes = [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]
            landmarks = self.fa.get_landmarks(gray, face_boxes)
            return landmarks
        elif len(faces) is not 0:
            landmarks = []
            for face in faces:
                detected_landmarks = self.fa(gray, face).parts()
                landmarks.append([[p.x, p.y] for p in detected_landmarks])
            return landmarks
        else:
            print("No face detected")
            return None

    def draw_landmark(self, image, faces, landmarks = None):
        black_image = np.zeros(image.shape, np.uint8)
        for face in faces:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), color=(0, 255, 255))

        # perform if there is a landmark
        if landmarks is not None:
            for landmark in landmarks:
                jaw = self.reshape_for_polyline(landmark[0:17])
                left_eyebrow = self.reshape_for_polyline(landmark[22:27])
                right_eyebrow = self.reshape_for_polyline(landmark[17:22])
                nose_bridge = self.reshape_for_polyline(landmark[27:31])
                lower_nose = self.reshape_for_polyline(landmark[30:35])
                left_eye = self.reshape_for_polyline(landmark[42:48])
                right_eye = self.reshape_for_polyline(landmark[36:42])
                outer_lip = self.reshape_for_polyline(landmark[48:60])
                inner_lip = self.reshape_for_polyline(landmark[60:68])

                color = (255, 255, 255)
                thickness = 1

                cv2.polylines(black_image, [jaw[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [left_eyebrow[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [right_eyebrow[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [nose_bridge[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [lower_nose[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [left_eye[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [right_eye[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [outer_lip[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [inner_lip[:, :, :2]], True, color, thickness)

            return cv2.add(image, black_image)
        else:
            print("No landmark detected")
            return image

image = cv2.imread("../mail.naver.com.png")
image = cv2.resize(image, dsize = (0,0), fx=0.3, fy=0.3)
print(image.shape)
FD = FaceDetector()
face_box = FD.detect_anim(image)

print(image.shape)
# FD = FaceDetector()
# face_box = FD.detect_face(image)
# left = 50
# top = 50
# right = 1200
# bottom = 700
# face_box = dlib.rectangle(left, top, right, bottom)
LE = LandmarkExtractor("../shape_predictor_68_face_landmarks.dat")
landmarks = LE.landmark_extractor(image, face_box)
cv2.imshow("detect", LE.draw_landmark(image, face_box, landmarks))
cv2.waitKey()
