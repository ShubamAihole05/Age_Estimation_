import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
age_category = ['(0-3)','(4-10)','(24-37)','(38-43)', '(48-53)', '(60-100)']
model = load_model(r"C:\Users\SHUBAM\Downloads\vggAge\model_age2.h5")
model.summary()

frame_widht = 450
frame_height = 720

#padding=20
#bbox=[]

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    #bboxs=faceBox(faceNet,frame)
    while cap.isOpened():

        ret, frame = cap.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame,(450,720))
        #face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]


        try:
            # if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces =  face_detection.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if faces.detections:
                print(type(faces.detections),len(faces.detections))
                for face_bonding in faces.detections:
                    location_data = face_bonding.location_data
                    bbox = location_data.relative_bounding_box
                    x =int( bbox.xmin *frame_widht) 
                    y =int( bbox.ymin * frame_height) -50
                    w =int( bbox.width *frame_widht + bbox.xmin * frame_widht)
                    h =int( bbox.height *frame_height + bbox.ymin * frame_height)
                    crop_face = frame[y:h, x:w]
                    crop_face = cv2.resize(crop_face/255,(128,128))
                    crop_face = np.expand_dims(crop_face, axis=0)
                    print(crop_face.shape)
                    result = model.predict(crop_face)
                    age_label = age_category[np.argmax(result)]

                    cv2.putText(frame,age_label, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,222,220),1)
                    cv2.rectangle(frame, (int(x),int(y)), (int(w),int(h)),(255,0,0,),1)


            cv2.imshow("Frame", frame)
        except Exception as e:
            raise
            print(e)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

