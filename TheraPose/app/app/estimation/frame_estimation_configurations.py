import customtkinter  as ctk
import tkinter as tk
from tkinter.colorchooser import askcolor
import numpy as np

import general.ui_images as ui_images
import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.create_scrollable_frame import CreateScrollableFrame

from estimation.algorithm_information import AlgorithmInformation

class FrameChooseConfigurations(CreateFrame):
    def __init__(self, master, text, number_of_points, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,1), arr=None), **kwargs)
        self.text=text
        self.number_of_points=number_of_points
        self.points_color=(255,0,0)
        self.connections_color=(0,255,0)
        self.var_select_all=ctk.IntVar(value=1)
        self.var_select_all.trace_add("write", self.select_all_trace_callback)
        self.var_list=[ctk.IntVar(value=1) for i in range(self.number_of_points)]
        
        n=8
        self.frame_check_boxes=CreateFrame(master=self, grid_frame=GridFrame(dim=(n, int(np.ceil(self.number_of_points/n))), arr=None))
        for i in range(self.number_of_points):
            self.frame_check_boxes.insert_element(cad_pos="{},{}".format(i-n*(i//n), i//n), element=FrameHelp.create_check_box(master=self.frame_check_boxes, text="{}".format(i), variable=self.var_list[i], onvalue=1, offvalue=0, width=50), padx=5, pady=5, sticky="w")

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text=self.text, size=14, weight="bold"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_check_box(master=self, text="Seleccionar todo", variable=self.var_select_all, onvalue=1, offvalue=0), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=self.frame_check_boxes, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self, text="Color de puntos", weight="bold", fg_color=utils.rgb_to_hex(color=self.points_color), command=lambda: self.button_select_color(button_type="points")), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_button(master=self, text="Color de conexiones", weight="bold", fg_color=utils.rgb_to_hex(color=self.connections_color), command=lambda: self.button_select_color(button_type="connections")), padx=5, pady=5, sticky="ew")

    def select_all_trace_callback(self, var, index, mode):
        for i in range(self.number_of_points): self.var_list[i].set(value=1 if self.var_select_all.get() else 0)

    def button_select_color(self, button_type):
        if button_type == "points":
            color=askcolor(title="Elige un color", initialcolor=utils.rgb_to_hex(color=self.points_color))
            if color[0] is not None:
                self.points_color=color[0]
                self.get_element(cad_pos="3,0").configure(fg_color=utils.rgb_to_hex(color=self.points_color))
        elif button_type == "connections":
            color=askcolor(title="Elige un color", initialcolor=utils.rgb_to_hex(color=self.connections_color))
            if color[0] is not None:
                self.connections_color=color[0]
                self.get_element(cad_pos="4,0").configure(fg_color=utils.rgb_to_hex(color=self.connections_color))

    def disable_configurations(self):
        self.get_element(cad_pos="1,0").configure(state=ctk.DISABLED)
        cad_pos_list=self.frame_check_boxes.elements.keys()
        for cad_pos in cad_pos_list:
            self.frame_check_boxes.get_element(cad_pos=cad_pos).configure(state=ctk.DISABLED)
        self.get_element(cad_pos="3,0").configure(state=ctk.DISABLED)
        self.get_element(cad_pos="4,0").configure(state=ctk.DISABLED)

class FrameAlgorithmItem(CreateScrollableFrame):
    def __init__(self, master, algorithm_information: AlgorithmInformation, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,2), arr=None), orientation="horizontal", **kwargs)
        self.algorithm_information=algorithm_information

        self.frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2 if self.algorithm_information.is_double_estimate else 1), arr=None))
        if self.algorithm_information.is_double_estimate:
            self.frame_container.insert_element(cad_pos="0,0", element=FrameChooseConfigurations(master=self.frame_container, text="Izquierda", number_of_points=self.algorithm_information.number_of_points), padx=5, pady=5, sticky="ew")
            self.frame_container.insert_element(cad_pos="0,1", element=FrameChooseConfigurations(master=self.frame_container, text="Derecha", number_of_points=self.algorithm_information.number_of_points), padx=5, pady=5, sticky="ew")
        else:
            self.frame_container.insert_element(cad_pos="0,0", element=FrameChooseConfigurations(master=self.frame_container, text="General", number_of_points=self.algorithm_information.number_of_points), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=ctk.CTkLabel(master=self, text="", image=ui_images.get_algorithm_image(name=self.algorithm_information.algorithm_name, width=300, height=300)), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,1", element=self.frame_container, padx=5, pady=5, sticky="ew")
    
    def disable_configurations(self):
        cad_pos_list=["0,0","0,1"] if self.algorithm_information.is_double_estimate else ["0,0"]
        for cad_pos in cad_pos_list:
            self.frame_container.get_element(cad_pos=cad_pos).disable_configurations()

class FrameEstimationConfigurations(CreateFrame):
    algorithm_information_list=[
        AlgorithmInformation(algorithm_name="mediapipe_pose_estimation", is_double_estimate=False, number_of_points=33, connections=[(0,4),(4,5),(5,6),(6,8),(0,1),(1,2),(2,3),(3,7),(9,10),(12,14),(14,16),(16,18),(18,20),(20,16),(16,22),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),(11,12),(12,24),(24,26),(26,28),(28,30),(30,32),(32,28),(11,23),(23,25),(25,27),(27,29),(29,31),(31,27),(23,24)]),
        AlgorithmInformation(algorithm_name="mediapipe_hand_estimation", is_double_estimate=True, number_of_points=21, connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)])
    ]

    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)

        self.frame_algorithms_list=CreateScrollableFrame(master=self, grid_frame=GridFrame(dim=(len(FrameEstimationConfigurations.algorithm_information_list),1), arr=None), height=500)
        for i in range(len(FrameEstimationConfigurations.algorithm_information_list)):
            self.frame_algorithms_list.insert_element(cad_pos="{},0".format(i), element=FrameAlgorithmItem(master=self.frame_algorithms_list, algorithm_information=FrameEstimationConfigurations.algorithm_information_list[i], height=450), padx=5, pady=5, sticky="ew")

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Seleccion de algoritmos", size=16, weight="bold", fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=self.frame_algorithms_list, padx=5, pady=5, sticky="nsew")

    def set_configurations_dict(self, configurations_dict):
        d1={}
        for i in range(len(FrameEstimationConfigurations.algorithm_information_list)):
            d1[FrameEstimationConfigurations.algorithm_information_list[i].algorithm_name]="{},0".format(i)
        for algorithm_name in d1.keys():            
            frame_algorithm_item: FrameAlgorithmItem=self.frame_algorithms_list.get_element(cad_pos=d1[algorithm_name])
            if frame_algorithm_item.algorithm_information.is_double_estimate:
                d2={"left": "0,0", "right": "0,1"}
                for key in d2.keys():
                    frame_algorithm_item.frame_container.get_element(cad_pos=d2[key]).var_select_all.set(value=0)
                if algorithm_name in configurations_dict:
                    if 'left' in configurations_dict[algorithm_name]:
                        if len(configurations_dict[algorithm_name]['left']['enabled_points']) == frame_algorithm_item.algorithm_information.number_of_points:
                            frame_algorithm_item.frame_container.get_element(cad_pos=d2['left']).var_select_all.set(value=1)
                        else:
                            for p in configurations_dict[algorithm_name]['left']['enabled_points']:
                                frame_algorithm_item.frame_container.get_element(cad_pos=d2['left']).var_list[p].set(value=1)
                    if 'right' in configurations_dict[algorithm_name]:
                        if len(configurations_dict[algorithm_name]['right']['enabled_points']) == frame_algorithm_item.algorithm_information.number_of_points:
                            frame_algorithm_item.frame_container.get_element(cad_pos=d2['right']).var_select_all.set(value=1)
                        else:
                            for p in configurations_dict[algorithm_name]['right']['enabled_points']:
                                frame_algorithm_item.frame_container.get_element(cad_pos=d2['right']).var_list[p].set(value=1)
            else:
                frame_algorithm_item.frame_container.get_element(cad_pos="0,0").var_select_all.set(value=0)
                if algorithm_name in configurations_dict:
                    if len(configurations_dict[algorithm_name]['enabled_points']) == frame_algorithm_item.algorithm_information.number_of_points:
                        frame_algorithm_item.frame_container.get_element(cad_pos="0,0").var_select_all.set(value=1)
                    else:
                        for p in configurations_dict[algorithm_name]['enabled_points']:
                            frame_algorithm_item.frame_container.get_element(cad_pos="0,0").var_list[p].set(value=1)

    def get_configurations_dict(self):
        configurations_dict={}
        for i in range(len(FrameEstimationConfigurations.algorithm_information_list)):
            frame_algorithm_item: FrameAlgorithmItem=self.frame_algorithms_list.get_element(cad_pos="{},0".format(i))
            algorithm_name=frame_algorithm_item.algorithm_information.algorithm_name
            configurations_dict[algorithm_name]={"is_double_estimate": frame_algorithm_item.algorithm_information.is_double_estimate, "classes": {"camera_1": AlgorithmInformation.get_corresponding_algorithm_class(algorithm_name=algorithm_name), "camera_2": AlgorithmInformation.get_corresponding_algorithm_class(algorithm_name=algorithm_name)}}
            if frame_algorithm_item.algorithm_information.is_double_estimate:
                d={"left": "0,0", "right": "0,1"}
                for key in d.keys():
                    enabled_points=np.where([frame_algorithm_item.frame_container.get_element(cad_pos=d[key]).var_list[i].get() for i in range(len(frame_algorithm_item.frame_container.get_element(cad_pos=d[key]).var_list))])[0].tolist()
                    if key not in configurations_dict[algorithm_name]: 
                        configurations_dict[algorithm_name][key]={}
                    configurations_dict[algorithm_name][key]["enabled_points"]=enabled_points
                    configurations_dict[algorithm_name][key]["connections"]=frame_algorithm_item.algorithm_information.get_new_connections(enabled_points=enabled_points)
                    configurations_dict[algorithm_name][key]["points_color"]=frame_algorithm_item.frame_container.get_element(cad_pos=d[key]).points_color
                    configurations_dict[algorithm_name][key]["connections_color"]=frame_algorithm_item.frame_container.get_element(cad_pos=d[key]).connections_color
                if len(configurations_dict[algorithm_name]['left']['enabled_points']) == 0 and len(configurations_dict[algorithm_name]['right']['enabled_points']) == 0:
                    del configurations_dict[algorithm_name]
            else:
                enabled_points=np.where([frame_algorithm_item.frame_container.get_element(cad_pos="0,0").var_list[i].get() for i in range(len(frame_algorithm_item.frame_container.get_element(cad_pos="0,0").var_list))])[0].tolist()
                configurations_dict[algorithm_name]["enabled_points"]=enabled_points
                configurations_dict[algorithm_name]["connections"]=frame_algorithm_item.algorithm_information.get_new_connections(enabled_points=enabled_points)
                configurations_dict[algorithm_name]["points_color"]=frame_algorithm_item.frame_container.get_element(cad_pos="0,0").points_color
                configurations_dict[algorithm_name]["connections_color"]=frame_algorithm_item.frame_container.get_element(cad_pos="0,0").connections_color
                if len(enabled_points) == 0:
                    del configurations_dict[algorithm_name]
        return configurations_dict
    
    def disable_configurations(self):
        d1={}
        for i in range(len(FrameEstimationConfigurations.algorithm_information_list)):
            d1[FrameEstimationConfigurations.algorithm_information_list[i].algorithm_name]="{},0".format(i)
        for algorithm_name in d1.keys():            
            frame_algorithm_item: FrameAlgorithmItem=self.frame_algorithms_list.get_element(cad_pos=d1[algorithm_name])
            frame_algorithm_item.disable_configurations()