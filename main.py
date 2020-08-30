import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# Capture images
#img = cv2.imread('HP.jpg')
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_coordinates = trained_face_data.detectMultiScale(img_gray)

# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

# cv2.imshow("Face Detector", img)
# cv2.waitKey()


# Capture video

webcam = cv2.VideoCapture(0)
while True:
    frame_read, frame = webcam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_frame)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.imshow("Face detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
