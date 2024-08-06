from threading import Thread
import time
import copy
import numpy as np

from general.thread_camera import ThreadCamera
from general.template_thread import TemplateThread
import general.utils as utils

from calibration.calibration_functions import CalibrationFunctions

class ThreadEstimationCalculation(Thread, TemplateThread):
    def __init__(self, thread_camera_1: ThreadCamera, thread_camera_2: ThreadCamera, Q1, Q2, configurations_dict, estimations_dict_callback, **kwargs):
        Thread.__init__(self, **kwargs)
        TemplateThread.__init__(self)
        self.thread_camera_1=thread_camera_1
        self.thread_camera_2=thread_camera_2
        self.Q1=Q1
        self.Q2=Q2
        self.configurations_dict=configurations_dict
        self.estimations_dict_callback=estimations_dict_callback
        
    def run(self):
        self.loop_estimation_calculation()
        print("End ThreadEstimationCalculation")

    def coordinate_converter(self, vws):
        # Conversion de sistemas de coordenadas (plano de frente a plano acostado)
        vws_=(utils.RzRyRx(psi=-np.pi/2, theta=0, phi=np.pi)@vws.T).T
        return vws_

    def loop_estimation_calculation(self):
        while not self.event_kill_thread.is_set():
            calibration_information_1=copy.deepcopy(self.thread_camera_1.calibration_information)
            calibration_information_2=copy.deepcopy(self.thread_camera_2.calibration_information)
            frame_1=copy.deepcopy(self.thread_camera_1.frame)
            frame_2=copy.deepcopy(self.thread_camera_2.frame)
            if calibration_information_1 is None or calibration_information_2 is None:
                self.kill_thread()
                return
            
            estimations_dict={"algorithms": {}, "frames": {}}
            for algorithm_name in self.configurations_dict.keys():
                estimations_dict["algorithms"][algorithm_name]={}
                algorithm_class_1=self.configurations_dict[algorithm_name]["classes"]["camera_1"]
                algorithm_class_2=self.configurations_dict[algorithm_name]["classes"]["camera_2"]
                v3ps1=algorithm_class_1.get_points_2D(frame=frame_1)
                v3ps2=algorithm_class_2.get_points_2D(frame=frame_2)
                if self.configurations_dict[algorithm_name]["is_double_estimate"]:
                    estimations_dict["algorithms"][algorithm_name]={"left": {"v3ps1": None, "v3ps2": None, "vws": None}, "right": {"v3ps1": None, "v3ps2": None, "vws": None}}
                    if v3ps1 is not None:
                        for key in v3ps1.keys(): 
                            estimations_dict["algorithms"][algorithm_name][key]["v3ps1"]=v3ps1[key]
                            frame_1=CalibrationFunctions.draw_algorithm_points_on_frame(frame=frame_1, v3ps=v3ps1[key], enabled_points=self.configurations_dict[algorithm_name][key]["enabled_points"], connections=self.configurations_dict[algorithm_name][key]["connections"], points_color=self.configurations_dict[algorithm_name][key]["points_color"], connections_color=self.configurations_dict[algorithm_name][key]["connections_color"], create_copy=False)
                    if v3ps2 is not None:
                        for key in v3ps2.keys(): 
                            estimations_dict["algorithms"][algorithm_name][key]["v3ps2"]=v3ps2[key]
                            frame_2=CalibrationFunctions.draw_algorithm_points_on_frame(frame=frame_2, v3ps=v3ps2[key], enabled_points=self.configurations_dict[algorithm_name][key]["enabled_points"], connections=self.configurations_dict[algorithm_name][key]["connections"], points_color=self.configurations_dict[algorithm_name][key]["points_color"], connections_color=self.configurations_dict[algorithm_name][key]["connections_color"], create_copy=False)
                    if estimations_dict["algorithms"][algorithm_name]['left']['v3ps1'] is not None and estimations_dict["algorithms"][algorithm_name]['left']['v3ps2'] is not None:
                        vws=CalibrationFunctions.get_3D_points(K1=calibration_information_1.K, Q1=self.Q1, q1=calibration_information_1.q, K2=calibration_information_2.K, Q2=self.Q2, q2=calibration_information_2.q, v3ps1=estimations_dict["algorithms"][algorithm_name]['left']['v3ps1'], v3ps2=estimations_dict["algorithms"][algorithm_name]['left']['v3ps2'])
                        estimations_dict["algorithms"][algorithm_name]['left']["vws"]=self.coordinate_converter(vws=vws)
                    if estimations_dict["algorithms"][algorithm_name]['right']['v3ps1'] is not None and estimations_dict["algorithms"][algorithm_name]['right']['v3ps2'] is not None:
                        vws=CalibrationFunctions.get_3D_points(K1=calibration_information_1.K, Q1=self.Q1, q1=calibration_information_1.q, K2=calibration_information_2.K, Q2=self.Q2, q2=calibration_information_2.q, v3ps1=estimations_dict["algorithms"][algorithm_name]['right']['v3ps1'], v3ps2=estimations_dict["algorithms"][algorithm_name]['right']['v3ps2'])
                        estimations_dict["algorithms"][algorithm_name]['right']["vws"]=self.coordinate_converter(vws=vws)
                else:
                    estimations_dict["algorithms"][algorithm_name]={"v3ps1": v3ps1, "v3ps2": v3ps2, "vws": None}
                    frame_1=CalibrationFunctions.draw_algorithm_points_on_frame(frame=frame_1, v3ps=v3ps1, enabled_points=self.configurations_dict[algorithm_name]["enabled_points"], connections=self.configurations_dict[algorithm_name]["connections"], points_color=self.configurations_dict[algorithm_name]["points_color"], connections_color=self.configurations_dict[algorithm_name]["connections_color"], create_copy=False)
                    frame_2=CalibrationFunctions.draw_algorithm_points_on_frame(frame=frame_2, v3ps=v3ps2, enabled_points=self.configurations_dict[algorithm_name]["enabled_points"], connections=self.configurations_dict[algorithm_name]["connections"], points_color=self.configurations_dict[algorithm_name]["points_color"], connections_color=self.configurations_dict[algorithm_name]["connections_color"], create_copy=False)
                    if v3ps1 is not None and v3ps2 is not None:
                        vws=CalibrationFunctions.get_3D_points(K1=calibration_information_1.K, Q1=self.Q1, q1=calibration_information_1.q, K2=calibration_information_2.K, Q2=self.Q2, q2=calibration_information_2.q, v3ps1=v3ps1, v3ps2=v3ps2)
                        estimations_dict["algorithms"][algorithm_name]["vws"]=self.coordinate_converter(vws=vws)
                estimations_dict["algorithms"][algorithm_name]["is_double_estimate"]=self.configurations_dict[algorithm_name]["is_double_estimate"]
            estimations_dict["frames"]["frame_1"]=frame_1
            estimations_dict["frames"]["frame_2"]=frame_2

            self.estimations_dict_callback(estimations_dict=estimations_dict)

            time.sleep(0.01)


