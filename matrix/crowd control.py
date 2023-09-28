import cv2
import numpy as np


cap= [r"E:\project\video\crowded scene.mp4"]  #source


# Crowd size
crowd_threshold = 400

def count_crowd(frame, bg_subtractor, nms_threshold=0.3):
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #NMS
    filtered_contours = cv2.dnn.NMSBoxes(
        [cv2.boundingRect(c) for c in contours],
        [1.0] * len(contours),
        0.0,
        nms_threshold
    )

    return len(filtered_contours)


video_captures = [cv2.VideoCapture(source) for source in cap]

while True:
    crowd_sizes = []

    for i, cap in enumerate(video_captures):
        ret, frame = cap.read()
        if not ret:
            break

        crowd_size = count_crowd(frame, background_subtractors[i])
        crowd_sizes.append(crowd_size)

        cv2.putText(frame, f"Crowd Size: {crowd_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if crowd_size > crowd_threshold:
            cv2.putText(frame, "Crowd Alert!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(f"Camera {i+1}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in video_captures:
    cap.release()
cv2.destroyAllWindows()
