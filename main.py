import numpy as np

import cv2
import face_recognition

# importing our image
# STEP 1
imgElon = face_recognition.load_image_file('images/elon1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)  # changing it into rgb channel

imgTest = face_recognition.load_image_file('images/elon2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# STEP - 2
# Finding the faces in the image then finding their encoding as well

# detecting face
faceLoc = face_recognition.face_locations(imgElon)[0]  # This returns the top,left,right,bottom
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # I
# detecting face in test image
faceLocTest = face_recognition.face_locations(imgTest)[0]  # This returns the top,left,right,bottom
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)  # I

# Step -3 Comparing these faces and
# we are using linear svm in backend to find out whether they match or not
# we have a list of encodings of known face encodeElon that we are comparing it with encode Test
# It returns true if it found both the faces equals
results = face_recognition.compare_faces([encodeElon], encodeTest)

'''face_recognition.api. face_distance (face_encodings, face_to_compare)[source] Given a list of face encodings, 
compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you 
how similar the faces are. '''

face_dis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, face_dis)

cv2.putText(imgTest, f'{results} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('ELON MUSK', imgElon)
cv2.imshow('ELON Test', imgTest)
cv2.waitKey(0)
