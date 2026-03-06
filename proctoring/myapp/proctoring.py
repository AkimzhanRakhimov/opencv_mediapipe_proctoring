import cv2
import mediapipe as mp
import numpy as np

import time
from timeit import default_timer as timer
url="http://192.168.100.5:8080/video"

cap=cv2.VideoCapture(url)
time.sleep(2) 
mp_face=mp.solutions.face_mesh
face_mesh=mp_face.FaceMesh(refine_landmarks=True)
NOSE,LEFT_EYE,RIGHT_EYE=1,234,454

if not cap.isOpened():
    print("Cannot open IP camera!")
    exit()
else:
    print("okay")
left_eye_points=[33,133,160,158,144]
left_iris_points=[468,469,470,471,472]
right_eye_points=[362,263,385,387,373]
right_iris_points=[473,474,475,476,477]

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.5)

def detect_yaws_and_pitches(lm,w,h):
    nose=lm[NOSE]
    left_eye=lm[LEFT_EYE]
    right_eye=lm[RIGHT_EYE]
            
    nose_x=nose.x*w
    left_eye_x=left_eye.x*w
    right_eye_x=right_eye.x*w

    nose_y=nose.y*h
    left_eye_y=left_eye.y*h
    right_eye_y=right_eye.y*h

    face_center_x=(left_eye_x+right_eye_x)/2
    face_center_y=(left_eye_y+right_eye_y)/2

    if nose_x<face_center_x-30:
        text_1="Head is turned right!"
    elif nose_x>face_center_x+30:
        text_1="Head is turned left!"
    elif nose_y<face_center_y-25:
        text_1="Head is turned up!"
    elif nose_y>face_center_y+60:
        text_1="Head is turned down!"
    else:
        text_1=""
    return text_1
def gaze_detection(lm):
    left_eye_array=np.array([[lm[idx].x,lm[idx].y] for idx in left_eye_points])
    left_iris=np.array([[lm[idx].x,lm[idx].y] for idx in left_iris_points])
    left_eye_center=np.mean(left_eye_array,axis=0)
    left_iris_center=np.mean(left_iris,axis=0)
            
    right_eye_array=np.array([[lm[idx].x,lm[idx].y] for idx in right_eye_points])
    right_iris=np.array([[lm[idx].x,lm[idx].y] for idx in right_iris_points])
    right_eye_center=np.mean(right_eye_array,axis=0)
    right_iris_center=np.mean(right_iris,axis=0)
    left_eye_top=np.mean([lm[159].y,lm[145].y])
    left_eye_bottom=np.mean([lm[33].y,lm[133].y])
    left_eye_middle=(left_eye_bottom+left_eye_top)/2
    eye_height=left_eye_bottom-left_eye_top
    rel_pos=(left_iris_center[1]-left_eye_middle)/eye_height
    rel_pos=np.clip(rel_pos,0,1)
    dx=((left_eye_center[0]-left_iris_center[0])+(right_eye_center[0]-right_iris_center[0]))/2
    dy=((left_eye_center[1]-left_iris_center[1])+(right_eye_center[1]-right_iris_center[1]))/2
    text_2=""
    if dx>0.008:
        text_2+="Looking right "
    elif dx<-0.008:
        text_2+="Looking left "
    else:
        text_2+=""
    if dy>0.004  :
        text_2+="Looking up"
    # elif dy<-0.004:
    #     text_2="Looking down"
    # if rel_pos>-1.15:
    #     text_2="Looking up"
    elif rel_pos>0.15:
        text_2+="Looking down"
    else:
        text_2+=""
    return text_2


def generate_frames():
    while True:
        
        ret,frame=cap.read()

        
        if not ret:
            print("Camera hasn`t been found")
            break

        h,w,_=frame.shape
        
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=face_mesh.process(rgb)
        results_hands=hands.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm=face_landmarks.landmark
                
                cv2.putText(
                    frame,
                    ("Student present"),
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )
                cv2.putText(
                    frame,
                    (detect_yaws_and_pitches(lm=lm,w=w,h=h)),
                    (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2
                )
                cv2.putText(
                    frame,
                    (gaze_detection(lm=lm)),
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (120,255,0),
                    2
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_CONTOURS
                )
            
        else:
            cv2.putText(
                frame,
                "No face!",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,255),
                2
            )

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                cv2.putText(
                    frame,
                    ("Hands detected"),
                    (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,0),
                    2
                )

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield (b'--frame\r\n'
               b'content-type:image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1)==27:
            break

cap.release()
cv2.destroyAllWindows()