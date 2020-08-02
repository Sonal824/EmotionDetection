# Importing OpenCV library
import cv2

# Include haar-cascades classifiers for face eye and smile
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Detect smile from live feed i.e webcam
# faces = faceCascade.detectMultiScale(gray,1.3,5)

# For each face detect smileys 
def detect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # x and y is coordinates of upper left corner w and h is width and height of frame
    for (x, y , w, h) in faces:
        cv2.rectangle(frame, (x,y), ((x + w), (y + h)), (255, 0, 0), 2)
        regionOfInterestGray = gray[y:y + h, x:x + w]
        regionOfInterestColor = frame[y:y + h, x:x + w]
        smiles = smileCascade.detectMultiScale(regionOfInterestGray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(regionOfInterestColor, (sx, sy), ( (sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame

# Main function to trigger detect
captureVideo = cv2.VideoCapture(0)
while True:
    # Read frames
    _, frame = captureVideo.read()

    # Capture image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Call detect function
    canvas = detect(gray, frame)

    # Display result
    cv2.imshow('Video', canvas)

    # Close program on click of q key
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# After processing release capture
captureVideo.release()
cv2.destroyAllWindows()