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


@app.route('/caps.html')
def cap():
    return render_template('caps.html')


@app.route('/index.html')
def index():
    return render_template('index.html')

# All available eyewears
@app.route('/specs')
def specs():
    apply_goggles(specs.__name__)
    return render_template('index.html')

@app.route('/glass1')
def glass1():
    apply_goggles(glass1.__name__)
    return render_template('index.html')

@app.route('/blueGlass')
def blueGlass():
    apply_goggles(blueGlass.__name__)
    return render_template('index.html')

@app.route('/orangeGlass')
def orangeGlass():
    apply_goggles(orangeGlass.__name__)
    return render_template('index.html')


# All available head wears
@app.route('/cap1')
def cap1():
    apply_cap(cap1.__name__)
    return render_template('caps.html')

@app.route('/cap2')
def cap2():
    apply_cap(cap2.__name__)
    return render_template('caps.html')

@app.route('/crown')
def crown():
    apply_cap(crown.__name__)
    return render_template('caps.html')

@app.route('/headbow')
def headbow():
    apply_cap(headbow.__name__)
    return render_template('caps.html')


def image_resize(img, req_width):
    r = float(req_width) / img.shape[1]
    dim = (req_width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def apply_goggles(name):
    cap = cv2.VideoCapture(0)


    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        print(f"{dir_path}" + fr"\static\{name}.png")
        glasses = cv2.imread(f"{dir_path}" + fr"\static\{name}.png", -1)
        print(glasses)

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


def apply_cap(name):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        print(f"{dir_path}" + fr"\images\{name}.png")
        cap_img = cv2.imread(f"{dir_path}" + fr"\static\{name}.png", -1)
        print(cap_img)

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
                            frame[y + i - (h // 2), x + j] = cap_img[i, j]

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


if __name__ == "__main__":
    app.run(debug=True)

