import numpy as np
import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.frame_3D_graphic import Frame3DGraphic

class FrameEstimation3DGraphic(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.var_points=ctk.IntVar(value=1)
        self.var_connections=ctk.IntVar(value=1)

        self.frame_3D_graphic=Frame3DGraphic(master=self, width=400, height=400)
        self.frame_3D_graphic.graphic_configuration(title="Estimaciones 3D (unidades: cm)")
        self.frame_3D_graphic.enable_fixed_size()

        self.frame_check_boxes=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        self.frame_check_boxes.insert_element(cad_pos="0,0", element=FrameHelp.create_check_box(master=self.frame_check_boxes, text="Puntos", variable=self.var_points, onvalue=1, offvalue=0), padx=5, pady=5, sticky="")
        self.frame_check_boxes.insert_element(cad_pos="1,0", element=FrameHelp.create_check_box(master=self.frame_check_boxes, text="Conexiones", variable=self.var_connections, onvalue=1, offvalue=0), padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=self.frame_3D_graphic, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=self.frame_check_boxes, padx=5, pady=5, sticky="")