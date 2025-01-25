import cv2

def draw_detections(frame, detections, names):
    """
    Draws bounding boxes around detected faces and key points on the frame.

    Parameters:
    - frame: The image frame from the webcam.
    - detections: A list of detection results, where each result contains
      the coordinates of the bounding box and key points.

    Returns:
    - The frame with drawn detections.
    """
    for i in range(len(detections)):
        # Draw bounding box
        top, right, bottom, left = detections[i]
        name = names[i]
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Draw key points
        #for point in detection['keypoints']:
        #    cv2.circle(frame, point, 5, (0, 0, 255), -1)

    return frame