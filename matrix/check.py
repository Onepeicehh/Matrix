import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Load YOLO model with pre-trained weights and config file
net = cv2.dnn.readNet(r"E:/project/yolov/yolov3.weights",r'E:\matrix\yolov\yolov3.cfg')

# Load class names
with open(r"E:/project/dataset/coco.names", "r", encoding="utf-8", errors="ignore") as f:
    classes = f.read().strip().split("\n")

#source
video_files = [r"E:/project/video/crowded scene.mp4",r"E:/project/video/boy alone.mp4"]  # Add more video file paths as needed

def apply_non_max_suppression(boxes, scores, score_threshold=0.7, iou_threshold=0.6):
    # Initialize the list of picked indexes
    picked_indexes = []

    # Iterate over the bounding boxes and their scores
    for i in range(len(boxes)):
        if scores[i] > score_threshold:
            picked_indexes.append(i)

            # Calculate the Intersection over Union (IoU) of the current box
            for j in range(i + 1, len(boxes)):
                if scores[j] > score_threshold:
                    x1 = max(boxes[i][0], boxes[j][0])
                    y1 = max(boxes[i][1], boxes[j][1])
                    x2 = min(boxes[i][2], boxes[j][2])
                    y2 = min(boxes[i][3], boxes[j][3])

                    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                    box_i_area = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1)
                    box_j_area = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1)
                    iou = intersection_area / float(box_i_area + box_j_area - intersection_area)

                    if iou > iou_threshold:
                        if scores[i] > scores[j]:
                            picked_indexes.remove(j)
                        else:
                            picked_indexes.remove(i)

    return picked_indexes

social_distance_threshold = 100  # Adjust(in pixels)
crowd_threshold = 15 # Adjust 
crime_detected = False

for video_file in video_files:
    # Load input video
    cap = cv2.VideoCapture(video_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #dimensions
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)

        #storage
        boxes = []
        confidences = []
        class_ids = []
        centers = []

        # Process 
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.6 and classes[class_id] == "person":
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centers.append((center_x, center_y))
                    
                elif confidence > 0.4 and classes[class_id] == "gun":  
                    crime_detected = True

        #NMS
        picked_indexes = apply_non_max_suppression(boxes, confidences)

        
        for i in picked_indexes:
            box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            label = f"{classes[class_id]}: {confidence:.2f}"
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #social distancing 
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))

                if distance < social_distance_threshold:
                   
                    cv2.line(frame, centers[i], centers[j], (0, 0, 255), 2)

        #people count
        people_count = len(picked_indexes)
        cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #trigger crowd alert
        if people_count > crowd_threshold:
            alert_text = "Crowd Alert!"
            alert_color = (0, 0, 255)  # Red color
            cv2.putText(frame, alert_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
        else:
            text = "Not crowded!"
            color = (0, 0, 255)
            cv2.putText(frame,text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2)
            
        #trigger crime alert
        if crime_detected:
            crime_alert_text = "Crime Detected!"
            crime_alert_color = (0, 0, 255)  # Red color
            cv2.putText(frame, crime_alert_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, crime_alert_color, 2)

        cv2.setWindowTitle("CCTV Surveillance", "CCTV Surveillance")
        cv2.imshow("CCTV Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()
