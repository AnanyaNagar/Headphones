import cv2
# import tkinter

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Video capturing object
cap = cv2.VideoCapture(0)
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Width of the video
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Height of the video

# images object
glasses = cv2.imread(f"{dir_path}\images\specs.png", -1)

# # Tkinter window object:
# window = tkinter.Tk()
# window.title("Dressing Room")
# canvas = tkinter.Canvas(window, width = width, height = height)
# canvas.pack()
# window.mainloop()

# classifiers of various parts:
eye_classifier = cv2.CascadeClassifier(f"{dir_path}\classifiers\eyes.xml")
face_classifier = cv2.CascadeClassifier(f"{dir_path}\classifiers\haarcascade_frontalface_default.xml")
fullbody_classifier = cv2.CascadeClassifier(f"{dir_path}\classifiers\haarcascade_fullbody.xml")
print("2nd part successful")

def image_resize(img, req_width):
    r = float(req_width) / img.shape[1]
    dim = (req_width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    if ret:
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for face in faces:
            x, y, w, h = face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            # eye classifier:
            eyes = eye_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            # plotting eyes:
            for eye in eyes:
                ex, ey, ew, eh = eye
                region_of_eyes = roi_gray[ey:ey+eh, ex:ex+ew]
                # cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 3)
                glasses = image_resize(glasses, ew)

                gh, gw, gc = glasses.shape
                # print(glasses.shape, eye, frame.shape)
                for i in range(gh):
                    for j in range(gw):
                        if glasses[i, j][3] != 0:
                            frame[ey+i+20, ex+j] = glasses[i, j]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # canvas.create_image(0, 0, image = frame, anchor = tkinter.NW)
        # window.after(15, main)
    cv2.imshow("dressing room(q-quit, c-click)", frame)

    # If 'q' key from he keyboard is pressed, save the image in the system
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
    if key == ord("c"):
        cv2.imwrite("MyImage.jpg", frame)

cap.release()
cv2.destroyAllWindows()

# print("Now executing")
# if __name__ == "__main__":
#     main()
