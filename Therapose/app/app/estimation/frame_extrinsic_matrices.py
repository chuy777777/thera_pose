import numpy as np
import customtkinter  as ctk

import general.ui_images as ui_images
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp

from calibration.patterns.frame_pattern_aruco_markers import FramePatternArucoMarkers
from calibration.patterns.frame_pattern_chessboard import FramePatternChessboard
from calibration.calibration_functions import CalibrationFunctions

class FrameExtrinsicMatrices(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(6, 1), arr=None), **kwargs)
        self.var_pattern=ctk.StringVar(value="Marcadores ArUco")
        
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Configuraciones para el calculo de las matrices extrinsecas", size=16, weight="bold", fg_color="coral"), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_option_menu(master=self, variable=self.var_pattern, values=["Marcadores ArUco", "Tablero de ajedrez"], command=self.selected_pattern), padx=5, pady=5)
        self.insert_element(cad_pos="2,0", element=FramePatternArucoMarkers(master=self), padx=5, pady=5, sticky="ew").hide_frame()
        self.insert_element(cad_pos="3,0", element=FramePatternChessboard(master=self), padx=5, pady=5, sticky="ew").hide_frame()

        self.selected_pattern(pattern="Marcadores ArUco")

    def selected_pattern(self, pattern):
        self.show_element(cad_pos="2,0") if pattern == "Marcadores ArUco" else self.hide_element(cad_pos="2,0")
        self.show_element(cad_pos="3,0") if pattern == "Tablero de ajedrez" else self.hide_element(cad_pos="3,0")

    def get_extrinsic_matrices(self, frame1, K1, q1, frame2, K2, q2):
        frame_pattern=self.get_element(cad_pos="2,0") if self.var_pattern.get() == "Marcadores ArUco" else self.get_element(cad_pos="3,0")
        v3ps1,vws1=frame_pattern.get_corresponding_points(frame=frame1)
        v3ps2,vws2=frame_pattern.get_corresponding_points(frame=frame2)
        if v3ps1 is not None and v3ps2 is not None:
            Q1=CalibrationFunctions.get_Q(K=K1, q=q1, v3ps=v3ps1, vws=vws1)
            Q2=CalibrationFunctions.get_Q(K=K2, q=q2, v3ps=v3ps2, vws=vws2)
            return Q1,Q2
        return None,None
