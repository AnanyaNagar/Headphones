import cv2
import os
from flask import Flask, render_template

dir_path = os.path.dirname(os.path.realpath(__file__))


# classifiers of various parts:
eye_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\eyes.xml")
face_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\haarcascade_frontalface_default.xml")
fullbody_classifier = cv2.CascadeClassifier(f"{dir_path}"+r"\classifiers\haarcascade_fullbody.xml")

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/kids.html')
def kds():
    return render_template('kids.html')

@app.route('/women.html')
def wmn():
    return render_template('womenj1.html')

@app.route('/men.html')
def mn():
    return render_template('men.html')

@app.route('/womenj1.html')
def wdress():
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

        glasses = cv2.imread(f"{dir_path}" + r"\images\specs.png", -1)

        if ret:
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for face in faces:
                x, y, w, h = face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                # eye classifier:
                eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                # plotting eyes:
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    region_of_eyes = roi_gray[ey:ey + eh, ex:ex + ew]
                    # cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 3)
                    glasses = image_resize(glasses, ew)

                    gh, gw, gc = glasses.shape
                    # print(glasses.shape, eye, frame.shape)
                    for i in range(gh):
                        for j in range(gw):
                            if glasses[i, j][3] != 0:
                                frame[ey + i + 20, ex + j] = glasses[i, j]

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
    return "Head back to the previous page"


@app.route('/men1.html')
def mdress():
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

        glasses = cv2.imread(f"{dir_path}" + r"\images\specs.png", -1)

        if ret:
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for face in faces:
                x, y, w, h = face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                # eye classifier:
                eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                # plotting eyes:
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    region_of_eyes = roi_gray[ey:ey + eh, ex:ex + ew]
                    # cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 3)
                    glasses = image_resize(glasses, ew)

                    gh, gw, gc = glasses.shape
                    # print(glasses.shape, eye, frame.shape)
                    for i in range(gh):
                        for j in range(gw):
                            if glasses[i, j][3] != 0:
                                frame[ey + i + 20, ex + j] = glasses[i, j]

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
    return "Head back to the previous page"


@app.route('/kids1.html')
def kdress():
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

        glasses = cv2.imread(f"{dir_path}" + r"\images\specs.png", -1)

        if ret:
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for face in faces:
                x, y, w, h = face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                # eye classifier:
                eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                # plotting eyes:
                for eye in eyes:
                    ex, ey, ew, eh = eye
                    region_of_eyes = roi_gray[ey:ey + eh, ex:ex + ew]
                    # cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 3)
                    glasses = image_resize(glasses, ew)

                    gh, gw, gc = glasses.shape
                    # print(glasses.shape, eye, frame.shape)
                    for i in range(gh):
                        for j in range(gw):
                            if glasses[i, j][3] != 0:
                                frame[ey + i + 20, ex + j] = glasses[i, j]

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
    return "Head back to the previous page"


if __name__ == "__main__":
    app.run(debug=True)

