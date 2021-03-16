import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_detector.detectMultiScale(grayscaled_img)
    
    for (x, y, w, h) in face_coordinates:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        the_face = frame[y:y+h, x:x+w]

        facegrayscaled_img = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        smiles = smile_detector.detectMultiScale(facegrayscaled_img, scaleFactor=1.7, minNeighbors=20)
        
        # for (x_, y_, w_, h_) in smiles:
            # cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=2,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


    cv2.imshow('why so serious?', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
cv2.destroyAllWindows()



print("success")