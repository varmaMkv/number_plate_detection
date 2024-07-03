from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2
import os
import csv

CONFIDENCE_THRESHOLD=0.4
COLOR=(0,255,0)

def detect_number_plates(image, model, display=False):
    start = time.time()

    detections = model.predict(image)[0].boxes.data
    print(detections)
    print(detections.shape)

    if detections.shape != torch.Size([0, 6]):
        boxes = []
        confidence = []
        for detection in detections:
            conf = detection[4].item()

            if float(conf) < CONFIDENCE_THRESHOLD:
                continue

            boxes.append(detection[:4])
            confidence.append(conf)
            
        print(f"{len(boxes)} Number plates have been detected.")
        number_plate_list = []

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])

            number_plate_list.append([xmin, ymin, xmax, ymax])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "Number Plate: {:.2f}%".format(confidence[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        if display:
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = number_plate_list[i]
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"Number plate {i}", number_plate)

        end = time.time()
        print(f"Time to detect the number plate: {(end - start) * 1000:.0f} milliseconds")

        return number_plate_list
    else:
        print("No Number Plate Have been detected")
        return []


if __name__ == "__main__":

    model = YOLO("/Vehicle-Number-Plate-Detection/runs/detect/train3/weights/best.pt")
    file_path="/Vehicle-Number-Plate-Detection/datasets/images/test/852.jpg"

    _, file_extension = os.path.splitext(file_path)

    # Check the file extension
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing the image...")

        image = cv2.imread(file_path)
        number_plate_list = detect_number_plates(image, model,
                                                 display=True)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    else:
        pass