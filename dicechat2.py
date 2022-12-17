import cv2
import numpy as np

# Load the templates for each dice value
one_template = cv2.imread('one.jpg', 0)
two_template = cv2.imread('two.jpg', 0)
three_template = cv2.imread('three.jpg', 0)
four_template = cv2.imread('four.jpg', 0)
five_template = cv2.imread('five.jpg', 0)
six_template = cv2.imread('six.jpg', 0)

# Set up the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop through each dice in the frame
    dice = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        dice_roi = gray[y:y+h, x:x+w]
        dice.append(dice_roi)

    # Loop through each dice ROI and determine the value
    for dice_roi in dice:
        # Apply threshold to binarize the image
        _, dice_roi = cv2.threshold(dice_roi, 200, 255, cv2.THRESH_BINARY)

        # Match the dice ROI against each template
        one_match = cv2.matchTemplate(dice_roi, one_template, cv2.TM_CCOEFF_NORMED)[0][0]
        two_match = cv2.matchTemplate(dice_roi, two_template, cv2.TM_CCOEFF_NORMED)[0][0]
        three_match = cv2.matchTemplate(dice_roi, three_template, cv2.TM_CCOEFF_NORMED)[0][0]
        four_match = cv2.matchTemplate(dice_roi, four_template, cv2.TM_CCOEFF_NORMED)[0][0]
        five_match = cv2.matchTemplate(dice_roi, five_template, cv2.TM_CCOEFF_NORMED)[0][0]
        six_match = cv2.matchTemplate(dice_roi, six_template, cv2.TM_CCOEFF_NORMED)[0][0]

        # Find the maximum match
        match = max(one_match, two_match, three_match, four_match, five_match, six_match)

        # Determine the dice value based on the maximum match
        if match == one_match:
            value = 1
        elif match == two_match:
            value = 2
        elif match == three_match:
            value = 3
        elif match == four_match:
            value = 4
        elif match == five_match:
            value = 5
        elif match == six_match: