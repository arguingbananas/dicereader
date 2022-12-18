import cv2
import numpy as np

# Load the template images for each dice value
template_1 = cv2.imread("dice_1.jpg", 0)
template_2 = cv2.imread("dice_2.jpg", 0)
template_3 = cv2.imread("dice_3.jpg", 0)
template_4 = cv2.imread("dice_4.jpg", 0)
template_5 = cv2.imread("dice_5.jpg", 0)
template_6 = cv2.imread("dice_6.jpg", 0)

# Set up the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find contours in the frame
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each dice in the frame
    dice = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        dice_roi = gray[y : y + h, x : x + w]
        dice.append(dice_roi)

    # Loop through each dice ROI and determine the value
    for dice_roi in dice:
        # Apply threshold to binarize the image
        _, dice_roi = cv2.threshold(dice_roi, 200, 255, cv2.THRESH_BINARY)

        # Match the dice ROI against each template
        one_match = cv2.matchTemplate(dice_roi, template_1, cv2.TM_CCOEFF_NORMED)[0][0]
        two_match = cv2.matchTemplate(dice_roi, template_2, cv2.TM_CCOEFF_NORMED)[0][0]
        three_match = cv2.matchTemplate(dice_roi, template_3, cv2.TM_CCOEFF_NORMED)[0][0]
        four_match = cv2.matchTemplate(dice_roi, template_4, cv2.TM_CCOEFF_NORMED)[0][0]
        five_match = cv2.matchTemplate(dice_roi, template_5, cv2.TM_CCOEFF_NORMED)[0][0]
        six_match = cv2.matchTemplate(dice_roi, template_6, cv2.TM_CCOEFF_NORMED)[0][0]

        # Find the maximum match
        match = max(
            one_match, two_match, three_match, four_match, five_match, six_match
        )

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
            value = 5

        # Print the dice value to the console
        print(f"Dice value: {value}")

    # Display the frame with bounding boxes
    cv2.imshow("frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()
