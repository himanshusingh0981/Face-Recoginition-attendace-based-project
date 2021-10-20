from datetime import datetime

import numpy as np
import cv2
import face_recognition
import os

path = 'images'
image = []
classNames = []
myList = os.listdir(path)  # This will return the list containing all the names of images in folder
for cl in myList:
    curr_image = cv2.imread(f'{path}/{cl}')  # cv2.imread() method loads an image from the specified file
    image.append(curr_image)
    classNames.append(os.path.splitext(cl)[0])


# This function finds the encodings of all images


def markAttendace(name):
    with open('Attend.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


def findEncodings(images):
    encodeList = []
    for img in images:
        imgElon = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert it into RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(image)
print('Encoding complete')

#  STEP - 3
#  Find the matches between the encoding but currently we don't have any image to map it
#  so we use webcam to find the image
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
            markAttendace(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

print(len(encodeListKnown))
