class CameraCapture:
    def __init__(self):
        self.capture = None

    def start_capture(self):
        import cv2
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            # Process the frame (this will be handled in the processor)
            # ...

    def stop_capture(self):
        if self.capture is not None:
            self.capture.release()
            cv2.destroyAllWindows()