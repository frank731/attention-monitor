import cv2
from detection.face_detector import FaceDetector
#from detection.pose_estimator import PoseEstimator
from analysis.attention_classifier import AttentionClassifier

from utils.visualization import draw_detections

def main():
    face_detector = FaceDetector()
    all_names = face_detector.load_reference_images("class_images")
    #pose_estimator = PoseEstimator()
    attention_classifier = AttentionClassifier()

        # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if success is False:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for face detection
        faces, names = face_detector.detect_faces(gray_frame, rgb_frame)
        extracted_faces = face_detector.extract_face_images(frame, faces)
                                                            
        for face in extracted_faces:
            attention_classifier.classify_attention(face)
        
        # Visualize results
        # (Assuming a function in visualization.py to draw results)
        draw_detections(frame, faces, names)

        # Display the frame
        cv2.imshow('Attention Monitor', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop capturing video
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
