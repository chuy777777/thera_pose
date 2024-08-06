import customtkinter  as ctk
import tkinter as tk
from tkinter.colorchooser import askcolor
import copy
import numpy as np

from estimation.algorithms.algorithm_media_pipe_hand_estimation import AlgorithmMediaPipeHandEstimation
from estimation.algorithms.algorithm_media_pipe_pose_estimation import AlgorithmMediaPipePoseEstimation

class AlgorithmInformation():
    def __init__(self, algorithm_name, is_double_estimate, number_of_points, connections):
        self.algorithm_name=algorithm_name
        self.is_double_estimate=is_double_estimate
        self.number_of_points=number_of_points
        self.connections=connections

    def get_new_connections(self, enabled_points):
        new_connections=[]
        M=np.zeros((self.number_of_points, self.number_of_points))
        for conn in self.connections:
            p1,p2=conn
            M[p1,p2]=1
            M[p2,p1]=1
        disabled_points=list(set([i for i in range(self.number_of_points)]).difference(set(enabled_points)))
        for p in disabled_points:
            M[:,p]=0 
            M[p,:]=0
        for i in  range(self.number_of_points):
            for j in range(self.number_of_points):
                if M[i,j] == 1 and (i,j) not in new_connections and (j,i) not in new_connections:
                    new_connections.append((i,j))
        return new_connections

    @staticmethod
    def get_corresponding_algorithm_class(algorithm_name):
        if algorithm_name == "mediapipe_pose_estimation":
            return AlgorithmMediaPipePoseEstimation()
        elif algorithm_name == "mediapipe_hand_estimation":
            return AlgorithmMediaPipeHandEstimation()

