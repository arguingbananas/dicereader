import cv2
import numpy as np

# Capture image or video from camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce high frequency noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Use Canny edge detection to identify edges in the image
    edges = cv2.Canny(blur, 50, 150)

    # Use Hough transform to detect circles in the image
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # Make sure circles were detected
    if circles is not None:
        # Convert circles to integer coordinates
        circles = np.round(circles[0, :]).astype("int")

        # Loop through each detected circle
        for (x, y, r) in circles:
            # Draw circle on image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

            # Crop image to just the dice
            dice = frame[y-r:y+r, x-r:x+r]

            # Use template matching to identify the value of the dice
            result = cv2.matchTemplate(dice, template, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # If the match is strong enough, display the value of the dice on the image
            if maxVal > 0.8:
                cv2.putText(frame, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow("Dice Tracking", frame)

    # Check if user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()