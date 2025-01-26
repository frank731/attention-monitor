import cv2
from analysis.attention_classifier import AttentionStatus, STATUS_TO_TEXT

def status_to_color(status: AttentionStatus) -> tuple:
    if status == AttentionStatus.Focused:
        return (0, 255, 0)
    elif status == AttentionStatus.Distracted:
        return (255, 0, 0)
    elif status == AttentionStatus.OnPhone:
        return (0, 0, 255)

def draw_detections(frame, detections, names, status):
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
        x, y, w, h = detections[i]
        name = names[i]
        # Draw a box around the face
        color = status_to_color(status[i])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y + h - 25), (x + w, y + h), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.7
        cv2.putText(frame, name, (x + 6, y + h - 6), font, font_size, (255, 255, 255), 1)
        status_text = STATUS_TO_TEXT[status[i]]
        cv2.putText(frame, status_text, (x + 6, y + h - 6 + 30), font, font_size, (255, 255, 255), 1)

        # Draw key points
        #for point in detection['keypoints']:
        #    cv2.circle(frame, point, 5, (0, 0, 255), -1)

    return frame
