import subprocess
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Install necessary packages using pip
def install_packages(packages):
    """
    Installs the given list of packages using pip.
    """
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

# List of packages to install
packages_to_install = ["opencv-python", "numpy", "torch", "Pillow", "ultralytics"]
install_packages(packages_to_install)

# Function to open a file dialog and select an image
def select_image_file():
    """
    Opens a dialog box for the user to select an image file.
    Returns the selected file path or None if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_types = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=file_types)
    root.destroy()

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None

# Perform object detection on the selected image
def perform_object_detection(image_path):
    """
    Performs object detection on the given image using MobileNet SSD.
    Returns the image, bounding boxes, and labels.
    """
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # Paths to the model files
    prototxt_path = 'MobileNetSSD_deploy.prototxt'
    model_path = 'MobileNetSSD_deploy.caffemodel'

    if not os.path.isfile(prototxt_path) or not os.path.isfile(model_path):
        print("Model files are missing. Ensure 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' are present.")
        return None, None, None

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or cannot be read.")
        return None, None, None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    final_boxes = []
    final_labels = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = "Product" if idx >= len(CLASSES) else CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            final_boxes.append([startX, startY, endX - startX, endY - startY])
            final_labels.append(label)
            print(f"Detected '{label}' with confidence {confidence:.2f} at [{startX}, {startY}, {endX - startX}, {endY - startY}]")

    return image, final_boxes, final_labels

# Create AR-like overlay on the image
def create_overlay(image_path, bounding_box, thumbnail_path, label, output_path="output_overlay.jpg"):
    """
    Creates an AR-like overlay for a given object in an image.
    """
    img = cv2.imread(image_path)
    thumbnail = cv2.imread(thumbnail_path)

    if img is None or thumbnail is None:
        print("Error: Image or thumbnail not found or cannot be read.")
        return

    x, y, w, h = bounding_box
    img_height, img_width = img.shape[:2]

    # Reduced scale factor to make the overlay box smaller
    scale_factor = min(img_width, img_height) / 400  # Reduced from 300 to 400 for better fitting
    box_padding = int(15 * scale_factor)  # Adjusted padding
    thumbnail_height = int(80 * scale_factor)  # Adjusted thumbnail height
    radius = int(20 * scale_factor)

    # Resize the thumbnail to fit inside the overlay box
    aspect_ratio = thumbnail.shape[1] / thumbnail.shape[0]
    thumbnail_width = int(thumbnail_height * aspect_ratio)
    thumbnail = cv2.resize(thumbnail, (thumbnail_width, thumbnail_height))

    # Define dimensions for the overlay box
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.0 * scale_factor, 2)
    box_width = thumbnail_width + text_size[0] + 4 * box_padding
    box_height = thumbnail_height + 2 * box_padding

    # Determine the position of the overlay box (ensure it stays within image boundaries)
    box_x = x + w + int(15 * scale_factor)  # Place it to the right of the object
    box_y = max(y - box_height // 2, 0)  # Center vertically on the object

    # Adjust position to ensure the overlay box stays within image boundaries
    if box_x + box_width > img_width:
        box_x = x - box_width - int(15 * scale_factor)  # Place to the left if it overflows
    if box_x < 0:  # Ensure it doesn't go off-screen to the left
        box_x = 0

    if box_y + box_height > img_height:
        box_y = img_height - box_height - 10
    if box_y < 0:
        box_y = 0

    # Draw the blue rounded rectangular overlay box
    overlay = img.copy()
    draw_rounded_rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (249, 213, 69), radius=radius, thickness=-1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Place the thumbnail inside the box
    thumbnail_y = box_y + box_padding
    thumbnail_x = box_x + box_padding

    # Ensure the thumbnail fits within the image boundaries
    if thumbnail_y + thumbnail_height > img_height or thumbnail_x + thumbnail_width > img_width:
        thumbnail_height = min(thumbnail_height, img_height - thumbnail_y)
        thumbnail_width = min(thumbnail_width, img_width - thumbnail_x)
        thumbnail = cv2.resize(thumbnail, (thumbnail_width, thumbnail_height))

    img[thumbnail_y:thumbnail_y + thumbnail_height, thumbnail_x:thumbnail_x + thumbnail_width] = thumbnail

    # Add the label text beside the thumbnail
    text_x = thumbnail_x + thumbnail_width + box_padding
    text_y = thumbnail_y + int(thumbnail_height / 2)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.0 * scale_factor, (255, 255, 255), 1)

    # Draw the red bounding box around the detected object
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), max(2, int(2 * scale_factor)))  # Red box for the object

    # Draw a line connecting the bounding box to the overlay box
    line_start = (x + w, y + h // 2)
    line_end = (box_x, box_y + box_height // 2)
    if box_x < x:  # If the overlay box is to the left of the bounding box
        line_start = (x, y + h // 2)
    cv2.line(img, line_start, line_end, (0, 0, 0), 2)

    # Save and show the output image with the overlay
    cv2.imwrite(output_path, img)
    cv2.imshow("Overlay Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Helper function to draw rounded rectangles
def draw_rounded_rectangle(img, top_left, bottom_right, color, radius=20, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

# Main function to handle user input and call functions
def main():
    image_path = select_image_file()
    if image_path:
        image, bboxes, labels = perform_object_detection(image_path)

        if bboxes:
            thumbnail_path = select_image_file()
            if thumbnail_path:
                for bbox, lbl in zip(bboxes, labels):
                    create_overlay(image_path, bbox, thumbnail_path, lbl)

                print("Object detection and annotation completed successfully.")
            else:
                print("Thumbnail selection cancelled. Exiting the program.")
        else:
            print("No objects detected in the selected image.")
    else:
        print("Image selection cancelled. Exiting the program.")

if __name__ == "__main__":
    main()
