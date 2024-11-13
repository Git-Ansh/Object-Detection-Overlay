from tkinter import filedialog

import cv2
import numpy as np
import tkinter as tk

def draw_rounded_rectangle(img, top_left, bottom_right, color, radius=20, thickness=-1):
    """
    Draws a rounded rectangle on the image.

    Parameters:
    - img: The image to draw on.
    - top_left: Top-left corner coordinates (x, y).
    - bottom_right: Bottom-right corner coordinates (x, y).
    - color: Color of the rounded rectangle (B, G, R).
    - radius: Radius of the corners.
    - thickness: Thickness of the edges. -1 for filled rectangle.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw the four straight lines
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw the four corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

# def add_rounded_corners_to_thumbnail(thumbnail, radius=10):
#     """
#     Adds rounded corners to the thumbnail image.
#
#     Parameters:
#     - thumbnail: The thumbnail image.
#     - radius: Radius of the corners.
#
#     Returns:
#     - Thumbnail with rounded corners.
#     """
#     # Create a mask with rounded corners
#     mask = np.zeros_like(thumbnail, dtype=np.uint8)
#     height, width = thumbnail.shape[:2]
#     color = (255, 255, 255)  # White color for mask
#
#     # Draw rounded rectangle on the mask
#     cv2.rectangle(mask, (radius, 0), (width - radius, height), color, -1)
#     cv2.rectangle(mask, (0, radius), (width, height - radius), color, -1)
#     cv2.circle(mask, (radius, radius), radius, color, -1)
#     cv2.circle(mask, (width - radius, radius), radius, color, -1)
#     cv2.circle(mask, (radius, height - radius), radius, color, -1)
#     cv2.circle(mask, (width - radius, height - radius), radius, color, -1)
#
#     # Apply the mask to the thumbnail to create rounded corners
#     rounded_thumbnail = cv2.bitwise_and(thumbnail, mask)
#     return rounded_thumbnail

def create_overlay(image, bounding_box, thumbnail_path, label, output_path="output_overlay.jpg"):
    """
    Creates an AR-like overlay for a given object in an image.

    Parameters:
    - image: The main image where the object is detected.
    - bounding_box: The coordinates of the bounding box as a tuple (x, y, width, height).
    - thumbnail_path: Path to the thumbnail image for the object.
    - label: The name of the object.
    - output_path: Path where the output image with overlay will be saved.
    """

    # Read the main image and the thumbnail image
    img = cv2.imread(image)
    thumbnail = cv2.imread(thumbnail_path)

    if img is None:
        print("Error: Main image not found or cannot be read.")
        return

    if thumbnail is None:
        print("Error: Thumbnail image not found or cannot be read.")
        return

    # Extract bounding box coordinates
    x, y, w, h = bounding_box

    # Scale the box dimensions based on image size
    img_height, img_width = img.shape[:2]
    scale_factor = min(img_width, img_height) / 500  # Adjust scaling as needed
    box_padding = int(10 * scale_factor)  # Padding inside the box
    thumbnail_height = int(50 * scale_factor)  # Adjust thumbnail height with scaling
    radius = int(15 * scale_factor)  # Radius for rounded corners

    # Resize the thumbnail to fit inside the overlay box
    aspect_ratio = thumbnail.shape[1] / thumbnail.shape[0]
    thumbnail_width = int(thumbnail_height * aspect_ratio)
    thumbnail = cv2.resize(thumbnail, (thumbnail_width, thumbnail_height))

    # Add rounded corners to the thumbnail
    #thumbnail = add_rounded_corners_to_thumbnail(thumbnail, radius=10)

    # Define dimensions for the overlay box
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.7 * scale_factor, 2)
    box_width = thumbnail_width + text_size[0] + 3 * box_padding
    box_height = thumbnail_height + 2 * box_padding

    # Determine the position of the overlay box
    box_x = x + w + int(20 * scale_factor)  # Place the box to the right of the bounding box
    box_y = y

    # Ensure the box is within image bounds
    if box_x + box_width > img_width:
        box_x = x - box_width - int(20 * scale_factor)  # Place the box to the left if it exceeds bounds
    if box_y + box_height > img_height:
        box_y = img_height - box_height - 20
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
    img[thumbnail_y:thumbnail_y + thumbnail_height, thumbnail_x:thumbnail_x + thumbnail_width] = thumbnail

    # Add the label text beside the thumbnail
    text_x = thumbnail_x + thumbnail_width + box_padding
    text_y = thumbnail_y + thumbnail_height // 2 + int(7 * scale_factor)
    # Using Courier-like font (HERSHEY_COMPLEX is similar to Courier)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5 * scale_factor, (255, 255, 255), 1)

    # Draw the bounding box around the detected object
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw a line connecting the bounding box to the label box
    line_start = (x + w, y + h // 2)
    line_end = (box_x, box_y + box_height // 2)
    cv2.line(img, line_start, line_end, (1, 1, 1), 2)

    # Save and show the output image with the overlay
    cv2.imwrite(output_path, img)
    cv2.imshow("Overlay Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
def select_image_file():
    """
    Opens a dialog box for the user to select an image file.
    Returns the selected file path or None if no file is selected.
    """
    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Define file types
    file_types = [
        ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
        ("All Files", "*.*")
    ]

    # Open the file dialog
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=file_types)

    # Destroy the root window after selection
    root.destroy()

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None

image_path = select_image_file()
thumbnail_path = select_image_file()  # Replace with path to your thumbnail image
bounding_box = (50, 50, 10, 30)  # Example bounding box
label = "Bottle"

create_overlay(image_path, bounding_box, thumbnail_path, label)
