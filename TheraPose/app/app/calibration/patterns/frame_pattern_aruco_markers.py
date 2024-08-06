import numpy as np
import customtkinter  as ctk
import cv2

import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.frame_entry_validation import FrameEntryValidation, TextValidation, NumberValidation, GroupClassValidation

from calibration.calibration_functions import CalibrationFunctions, dict_aruco

class FramePatternArucoMarkers(CreateFrame):
    def __init__(self, master, aruco_type="DICT_7X7_50", aruco_id="0", square_size="145", **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs) 
        self.var_aruco_type=ctk.StringVar(value=aruco_type)               
        self.var_aruco_id=ctk.StringVar(value=aruco_id)
        self.var_square_size=ctk.StringVar(value=square_size)               

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_option_menu(master=self, variable=self.var_aruco_type, state="active", values=list(dict_aruco.keys()), command=self.selected_aruco_type), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_aruco_id, min_val=0, max_val=int(self.var_aruco_type.get().split("_")[2])-1), text_title="ID del marcador"), padx=5, pady=5)
        self.insert_element(cad_pos="2,0", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_square_size, min_val=10.0, max_val=500.0), text_title="Tamaño del cuadrado (en mm)"), padx=5, pady=5)
        
        self.group_class_validation=GroupClassValidation(list_class_validation=[
            self.get_element(cad_pos="1,0").class_validation, 
            self.get_element(cad_pos="2,0").class_validation
        ])

    def selected_aruco_type(self, aruco_type):
        self.get_element(cad_pos="1,0").class_validation.max_val=int(aruco_type.split("_")[2])-1
        self.get_element(cad_pos="1,0").callback_trace(None, None, None)

    def get_corresponding_points(self, frame):
        if self.group_class_validation.is_valid():
            aruco_type=self.var_aruco_type.get()
            aruco_id=int(self.var_aruco_id.get())
            square_size=float(self.var_square_size.get())
            v3ps=CalibrationFunctions.get_v3ps_from_aruco_marker_image(frame=frame, aruco_type=aruco_type, aruco_id=aruco_id)
            vws=CalibrationFunctions.get_vws_from_aruco_marker(square_size=square_size)
            return v3ps, vws
        else:
            return None, None
