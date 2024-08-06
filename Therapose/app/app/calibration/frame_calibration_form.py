import numpy as np
import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.frame_entry_validation import FrameEntryValidation, TextValidation, NumberValidation, GroupClassValidation

class FrameCalibrationForm(CreateFrame):
    def __init__(self, master, square_size="25.0", board_dimensions="9,6", number_images="3", timer_time="2.0", **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,2), arr=np.array([["0,0", "0,0"], ["1,0", "1,0"], ["2,0", "2,0"], ["3,0", "3,1"], ["4,0", "4,1"]])), **kwargs) 
        self.var_square_size=ctk.StringVar(value=square_size)               
        self.var_board_dimensions=ctk.StringVar(value=board_dimensions)
        self.var_number_images=ctk.StringVar(value=number_images)
        self.var_timer_time=ctk.StringVar(value=timer_time)                 

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Formulario de calibracion"), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, text="Ejemplo tablero de dimensiones (9,6) (horizontal,vertical)"), padx=5, pady=5)
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self, text="", img_name="board_96", img_width=300, img_height=200), padx=5, pady=5)
        self.insert_element(cad_pos="3,0", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_square_size, min_val=5.0, max_val=50.0), text_title="Tamaño de los cuadrados (en mm)"), padx=5, pady=5)
        self.insert_element(cad_pos="3,1", element=FrameEntryValidation(master=self, class_validation=TextValidation(var_entry=self.var_board_dimensions, max_length=5, regular_expression="[0-9]{1,2},[0-9]{1,2}"), text_title="Dimensiones del tablero"), padx=5, pady=5)
        self.insert_element(cad_pos="4,0", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_number_images, min_val=3, max_val=20), text_title="Numero de imagenes"), padx=5, pady=5)
        self.insert_element(cad_pos="4,1", element=FrameEntryValidation(master=self, class_validation=NumberValidation(var_entry=self.var_timer_time, min_val=2.0, max_val=10.0), text_title="Tiempo de captura (en s)"), padx=5, pady=5)
        
        self.group_class_validation=GroupClassValidation(list_class_validation=[
            self.get_element(cad_pos="3,0").class_validation, 
            self.get_element(cad_pos="3,1").class_validation,
            self.get_element(cad_pos="4,0").class_validation, 
            self.get_element(cad_pos="4,1").class_validation
        ])

    def get_fields(self):
        if self.group_class_validation.is_valid():
            square_size=float(self.var_square_size.get())
            board_dimensions=tuple(np.array(self.var_board_dimensions.get().split(",")).astype(int))
            number_images=int(self.var_number_images.get())
            timer_time=float(self.var_timer_time.get())
            return {"square_size": square_size, "board_dimensions": board_dimensions, "number_images": number_images, "timer_time": timer_time}
        else:
            return None
