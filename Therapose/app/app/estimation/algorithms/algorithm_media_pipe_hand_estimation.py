import numpy as np

import mediapipe as mp
mp_hands=mp.solutions.hands

"""
Algoritmo de deep learning para la estimacion de puntos caracteristicos de la mano.
"""
class AlgorithmMediaPipeHandEstimation():
    def __init__(self, **kwargs):
        self.hands=mp_hands.Hands(static_image_mode=False, model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.is_double_estimate=True

    # Devuelve: {"left": (m,2), "right": (m,2)}
    def get_points_2D(self, frame):
        if frame is not None:
            results = self.hands.process(frame)
            if results.multi_hand_landmarks is not None:

                l=["Left", "Right"]
                d=dict()
                hands_positions=[]
                for hand in results.multi_handedness:
                    hands_positions.append(list(set(l).difference(set([hand.classification[0].label])))[0])
                for i in range(len(results.multi_hand_landmarks)):
                    hand_landmarks=results.multi_hand_landmarks[i]
                    if hands_positions[i]=="Left": d['left']=hand_landmarks
                    elif hands_positions[i]=="Right": d['right']=hand_landmarks

                m=len(d[list(d.keys())[0]].landmark)
                d_points=dict()
                for k in list(d.keys()):
                    hand_landmarks=d[k]
                    points=np.zeros((m, 2))
                    for i in range(m):
                        x=hand_landmarks.landmark[i].x*frame.shape[1]
                        y=hand_landmarks.landmark[i].y*frame.shape[0]
                        points[i]=np.array([x,y])
                    d_points[k]=points
                return d_points

        return None