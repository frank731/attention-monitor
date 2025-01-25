import unittest
from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator

class TestDetection(unittest.TestCase):

    def setUp(self):
        self.face_detector = FaceDetector()
        self.pose_estimator = PoseEstimator()

    def test_face_detection(self):
        # Simulate a frame with faces
        frame = ...  # Replace with a test frame
        faces = self.face_detector.detect_faces(frame)
        self.assertIsInstance(faces, list)
        self.assertGreater(len(faces), 0, "No faces detected in the frame")

    def test_pose_estimation(self):
        # Simulate a frame with detected faces
        frame = ...  # Replace with a test frame
        faces = self.face_detector.detect_faces(frame)
        keypoints = self.pose_estimator.estimate_pose(frame, faces)
        self.assertIsInstance(keypoints, list)
        self.assertEqual(len(keypoints), len(faces), "Keypoints count does not match detected faces")

if __name__ == '__main__':
    unittest.main()