import cv2
import numpy as np

# Load YOLOv3 model trained on COCO dataset
net = cv2.dnn.readNetFromDarknet('C:/Users/NHI615/darknet/cfg/yolov3.cfg', 'C:/Users/NHI615/darknet/yolov3.weights')

# Define the classes that the YOLOv3 model can detect
classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

# Load input image
img = cv2.imread(r'C:\Users\NHI615\Documents\Hyundai-Grand-i10-Nios-200120231541.jpg')

# Convert image to 416x416 blob for input to YOLOv3
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

# Set input to the YOLOv3 network
net.setInput(blob)

# Get output layer names of the YOLOv3 network
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in [net.getUnconnectedOutLayers()]]

#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Run forward pass through the YOLOv3 network
outs = net.forward(output_layers)

# Initialize lists to store class IDs, confidence scores, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Loop over each detection from the YOLOv3 network
for out in outs:
    for detection in out:
        # Extract the class ID and confidence score for the detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Check if the detection is a vehicle and the confidence score is above 50%
        if confidence > 0.1 and classes[class_id] in classes:
            # Get the bounding box coordinates for the detection
            box = detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (center_x, center_y, width, height) = box.astype("int")
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))
            
            # Add the class ID, confidence score, and bounding box coordinates to the lists
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, int(width), int(height)])

# Apply non-maximum suppression to remove overlapping detections
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop over each detection after non-maximum suppression
for i in indices:
    if isinstance(i, list):
       i = i[0]

    # Get the bounding box coordinates and draw the bounding box and label on the image
    box = boxes[i]
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Vehicle Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
