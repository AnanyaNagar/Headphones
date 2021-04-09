import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# classifiers of various parts:
eye_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\eyes.xml")
face_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\haarcascade_frontalface_default.xml")
upperbody_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\haarcascade_upperbody.xml")

cap = cv2.VideoCapture(0)

def image_resize(img, req_width):
    r = float(req_width) / img.shape[1]
    dim = (req_width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cap_img = cv2.imread(f"{dir_path}"+r"\images\cap1.png", -1)

    if ret:
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for face in faces:
            x, y, w, h = face
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cap_img = image_resize(cap_img, w)
            #
            gh, gw, gc = cap_img.shape
            for i in range(gh):
                for j in range(gw):
                    if cap_img[i, j][3] != 0:
                        frame[y + i - (h//2), x + j] = cap_img[i, j]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow("dressing room(q-quit, c-click)", frame)

    # If 'q' key from he keyboard is pressed, save the image in the system
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
    if key == ord("c"):
        cv2.imwrite("MyImage.jpg", frame)

cap.release()
cv2.destroyAllWindows()