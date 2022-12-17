import cv2
import numpy as np

# Load the template images for each dice value
template_1 = cv2.imread('dice_1.jpg', 0)
template_2 = cv2.imread('dice_2.jpg', 0)
template_3 = cv2.imread('dice_3.jpg', 0)
template_4 = cv2.imread('dice_4.jpg', 0)
template_5 = cv2.imread('dice_5.jpg', 0)
template_6 = cv2.imread('dice_6.jpg', 0)

# Set up the video capture
capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    _, frame = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find contours in the frame
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and draw a bounding box around each dice
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Crop the dice out of the frame
        dice = gray[y:y+h, x:x+w]

        # Resize the dice to match the size of the templates
        dice = cv2.resize(dice, (100,100))

        # Initialize the maximum correlation value to 0
        max_corr = 0
        dice_value = 0

        # Loop through the templates and find the one with the highest correlation
        for i in range(1,7):
            template = eval(f'template_{i}')
            result = cv2.matchTemplate(dice, template, cv2.TM_CCOEFF_NORMED)
            corr = np.amax(result)
            if corr > max_corr:
                max_corr = corr
                dice_value = i

        # Print the dice value to the console
        print(f'Dice value: {dice_value}')

    # Display the frame with bounding boxes
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
capture.release()

# Close all windows
cv2.destroyAllWindows()