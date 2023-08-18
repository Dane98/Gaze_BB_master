import cv2

# Load the image
image = cv2.imread('/home/linlincheng/yolov5/data/images/img_box.jpg')
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply a threshold or other preprocessing if needed
# ...
# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
# Iterate over the contours
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # If the polygon has 4 vertices , assume it is a rectangle
    if len(approx) == 4:
    #Draw the rectangle on the image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        # Get the pixel positions of the rectangle
        x, y, w, h = cv2.boundingRect(approx)
        print(f"Top - left corner: ({x}, {y})")
        print(f"Bottom - right corner: ({x + w}, {y + h})")

# Display the image with the detected rectangles
cv2.imshow('Image with Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
