import os
import face_recognition

class FaceDetector:
    def __init__(self):
        self.face_encodings = []
        self.names = []

    def load_reference_images(self, reference_images_path):
        # Images need to follow name convention "firstname_lastname.ext"
        for filename in os.listdir(reference_images_path):
            name = " ".join(filename.split(".")[0].split("_")).title()

            image = face_recognition.load_image_file(os.path.join(reference_images_path, filename))
            face_encoding = face_recognition.face_encodings(image)[0]

            self.face_encodings.append(face_encoding)
            self.names.append(name)

    def detect_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.names[first_match_index]
            
            face_names.append(name)
        
        return face_locations, face_names

    def extract_face_images(self, frame, faces):
        face_images = []
        for face in faces:
            x, y, w, h = face
            face_images.append(frame[y:y+h, x:x+w])
        return face_images