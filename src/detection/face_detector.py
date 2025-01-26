import os
import face_recognition
import cv2
import np

class FaceDetector:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_encodings = []
        self.names = []

    def load_reference_images(self, reference_images_path):
        # Images need to follow name convention "firstname_lastname.ext"
        name_to_filepath = {}
        for filename in os.listdir(reference_images_path):
            name = " ".join(filename.split(".")[0].split("_")).title()

            image = face_recognition.load_image_file(os.path.join(reference_images_path, filename))
            face_encoding = face_recognition.face_encodings(image)[0]

            self.face_encodings.append(face_encoding)
            self.names.append(name)
            name_to_filepath[name] = os.path.join(reference_images_path, filename)
        return name_to_filepath

    def detect_faces(self, gray, frame):
        face_locations = self.face_classifier.detectMultiScale(gray) #face_recognition.face_locations(frame, model="hog")
        face_encodings = face_recognition.face_encodings(frame, self.convert_coordinate_format(face_locations))

        parsed_face_locations = []
        face_names = []
        for ind, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
            name = None

            face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names[best_match_index]
            
            if name is not None:
                parsed_face_locations.append(face_locations[ind])
                face_names.append(name)

        return parsed_face_locations, face_names

    def extract_face_images(self, frame, faces):
        face_images = []
        for face in faces:
            x, y, w, h = face
            face_images.append(frame[y:y+h, x:x+w])
        return face_images

    def convert_coordinate_format(self, coords):
        ret = []
        for coord in coords:
            top = coord[1]
            right = coord[0] + coord[2]
            bottom = coord[1] + coord[3]
            left = coord[0]
            ret.append((top, right, bottom, left))
        return ret
