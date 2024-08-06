import numpy as np

import mediapipe as mp
mp_pose=mp.solutions.pose

"""
Algoritmo de deep learning para la estimacion de puntos caracteristicos del cuerpo humano.
"""
class AlgorithmMediaPipePoseEstimation():
    def __init__(self, **kwargs):
        self.pose=mp_pose.Pose(static_image_mode=False)
        self.is_double_estimate=False

    # Devuelve: (m,2)
    def get_points_2D(self, frame):
        if frame is not None:
            results = self.pose.process(frame)
            if results.pose_landmarks is not None:

                m=len(results.pose_landmarks.landmark)
                points=np.zeros((m, 2))
                for i in range(m):
                    x=results.pose_landmarks.landmark[i].x*frame.shape[1]
                    y=results.pose_landmarks.landmark[i].y*frame.shape[0]
                    points[i]=np.array([x,y])
                return points

        return None