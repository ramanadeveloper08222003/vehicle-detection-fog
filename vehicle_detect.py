import cv2
import numpy as np
import winsound
import time

frequency = 1000  # Frequency of the beep sound in Hz
duration = 1000  # Duration of the beep sound in milliseconds


# Load the YOLO network
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Load the class names
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the Haar classifier for cars
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Load the video file
cap = cv2.VideoCapture('foggy_video.mp4')

while True:
    # Read each frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect the cars using Haar classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect the vehicles using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the YOLO detections
    confidences = []
    class_ids = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                class_ids.append(class_id)
                boxes.append([left, top, width, height])
                winsound.Beep(frequency, duration)

    # Draw rectangles around the detected cars and vehicles
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for i in range(len(boxes)):
        (x, y, w, h) = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Vehicle Detection', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
