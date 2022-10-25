import cv2
import numpy as np
from sklearn import cluster

params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        clustering = cluster.DBSCAN(eps=40, min_samples=1).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(r), (0, 0, 0), -1)

    # Overlay dice number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(
            frame,
            str(d[0]),
            (int(d[1] - textsize[0] / 2), int(d[2] + textsize[1] / 2)),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 0, 255),
            4,
        )


# Initialize a video feed
cap = cv2.VideoCapture(0)

# Initial zoom percentage
scale = 15

while True:
    # Grab the latest image from the video feed
    ret, frame = cap.read()

    # Borrowed from https://stackoverflow.com/questions/50870405/how-can-i-zoom-my-webcam-in-open-cv-python
    height, width, channels = frame.shape
    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)
    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY
    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    # We'll define these later
    blobs = get_blobs(resized_cropped)
    dice = get_dice_from_blobs(blobs)
    out_frame = overlay_info(resized_cropped, dice, blobs)

    cv2.imshow("Dice", resized_cropped)

    # Track zoom changes
    old_scale = scale

    res = cv2.waitKey(1)

    # Zoom in by 5's if user presses "a"
    if res & 0xFF == ord("a"):
        if (scale - 5) >= 5:
            scale -= 5  # -5
        else:
            scale = 5
    # Zoom out by 5's if user presses "z"
    if res & 0xFF == ord("z"):
        if (scale + 5) <= 50:
            scale += 5  # +5
        else:
            scale = 50
    # Zoom in by 1's if user presses "a"
    if res & 0xFF == ord("s"):
        if (scale - 1) >= 5:
            scale -= 1  # -1
        else:
            scale = 5
    # Zoom out by 1's if user presses "z"
    if res & 0xFF == ord("x"):
        if (scale + 1) <= 50:
            scale += 1  # +1
        else:
            scale = 50
    # Reset zoom to default if user presses "t"
    if res & 0xFF == ord("t"):
        scale = 15
    # Stop if the user presses "q"
    if res & 0xFF == ord("q"):
        break

    if old_scale != scale:
        print(scale)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
