import numpy as np
import customtkinter  as ctk
import cv2

import general.ui_images as ui_images
import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.frame_entry_validation import FrameEntryValidation, TextValidation, NumberValidation, GroupClassValidation

from calibration.calibration_functions import CalibrationFunctions

class FramePatternChessboard(CreateFrame):
    def __init__(self, master, square_size="25.0", board_dimensions="9,6", **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(4,2), arr=np.array([["0,0", "0,0"], ["1,0", "1,0"], ["2,0", "2,0"], ["3,0", "3,1"]])), **kwargs) 
        self.var_square_size=ctk.StringVar(value=square_size)               
        self.var_board_dimensions=ctk.StringVar(value=board_dimensions)

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Formulario de calibracion"), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, text="Ejemplo tablero de dimensiones (9,6) (horizontal,vertical)"), padx=5, pady=5)
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self, text="", img_name="board_96", img_width=400, img_height=300), padx=5, pady=5)
        self.insert_element(cad_pos="3,0", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_square_size, min_val=5.0, max_val=50.0), text_title="Tamaño de los cuadrados (en mm)"), padx=5, pady=5)
        self.insert_element(cad_pos="3,1", element=FrameEntryValidation(master=self, class_validation=TextValidation(var_entry=self.var_board_dimensions, max_length=5, regular_expression="[0-9]{1,2},[0-9]{1,2}"), text_title="Dimensiones del tablero"), padx=5, pady=5)
        
        self.group_class_validation=GroupClassValidation(list_class_validation=[
            self.get_element(cad_pos="3,0").class_validation, 
            self.get_element(cad_pos="3,1").class_validation,
        ])

    def get_corresponding_points(self, frame):
        if self.group_class_validation.is_valid():
            square_size=float(self.var_square_size.get())
            board_dimensions=tuple(np.array(self.var_board_dimensions.get().split(",")).astype(int))
            v3ps=CalibrationFunctions.get_v3ps_from_chessboard_image(frame=frame, board_dimensions=board_dimensions, fast=False)
            vws=CalibrationFunctions.get_vws_from_chessboard(board_dimensions=board_dimensions, square_size=square_size)
            return v3ps, vws
        else:
            return None, None
