import cv2
import numpy as np
from yolo_model import YOLO

# Create the YOLO model
yolo = YOLO(0.6, 0.5)

# Upload class names
file = "data/coco_classes.txt"
with open(file) as f:
    class_names = f.readlines()
all_classes = [c.strip() for c in class_names]

# List of colours (To prevent the rectangle colours from mixing after object detection)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Open camera
cap = cv2.VideoCapture(0)  # 0, 1, 2 gibi kamera indeksini belirtin

# Capture a frame from the camera
while True:
    ret, frame = cap.read()  
    if not ret:
        break

    # FPS calculation
    start_time = cv2.getTickCount()

    # Reshape and normalise frame size
    pimage = cv2.resize(frame, (416, 416))
    pimage = np.array(pimage, dtype="float32") / 255.0
    pimage = np.expand_dims(pimage, axis=0)

    # Make object detection
    boxes, classes, scores = yolo.predict(pimage, frame.shape)

    # Process detected objects
    for i, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
        x, y, w, h = box
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = max(0, np.floor(x + w + 0.5).astype(int))
        bottom = max(0, np.floor(y + h + 0.5).astype(int))

        # Assign a fixed colour for each object
        color = colors[i % len(colors)]

        cv2.rectangle(frame, (top, left), (right, bottom), color, 2)
        cv2.putText(frame, "{} {}".format(all_classes[cl], score), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # FPS calculation
    end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (end_time - start_time)

    # Print FPS value
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show image
    cv2.imshow("Yolo", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Switch off the camera and close the window
cap.release()
cv2.destroyAllWindows()


