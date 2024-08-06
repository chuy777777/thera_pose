import customtkinter  as ctk
import tkinter as tk
import numpy as np

import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.create_scrollable_frame import CreateScrollableFrame
from general.frame_camera_display import FrameCameraDisplay
from general.thread_camera import ThreadCamera

from estimation.frame_extrinsic_matrices import FrameExtrinsicMatrices
from estimation.thread_estimation_calculation import ThreadEstimationCalculation
from estimation.frame_estimation_3D_graphic import FrameEstimation3DGraphic
from estimation.coordinate_converter import CoordinateConverter
from estimation.frame_estimation_configurations import FrameEstimationConfigurations

class FrameEstimation(CreateScrollableFrame):
    def __init__(self, master, estimations_results_callback=None, verification_thread_estimation_calculation=None, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,2), arr=np.array([["0,0","0,0"],["1,0","1,1"],["2,0","2,0"],["3,0","3,0"],["4,0","4,0"]])), **kwargs)
        self.root=self.get_root()
        self.estimations_results_callback=estimations_results_callback
        self.verification_thread_estimation_calculation=verification_thread_estimation_calculation
        self.thread_camera_1: ThreadCamera=self.root.thread_camera_1
        self.thread_camera_2: ThreadCamera=self.root.thread_camera_2
        self.after_ms=100
        self.thread_estimation_calculation=None

        self.frame_cameras=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None), fg_color='lightpink')
        self.frame_camera_display_1=FrameCameraDisplay(master=self.frame_cameras, thread_camera=self.root.thread_camera_1, after_ms=10, scale_percent=50, show_calibration_information=True, editable=False, name="FrameCameraDisplayCalibration1")
        self.frame_camera_display_2=FrameCameraDisplay(master=self.frame_cameras, thread_camera=self.root.thread_camera_2, after_ms=10, scale_percent=50, show_calibration_information=True, editable=False, name="FrameCameraDisplayCalibration2")
        self.frame_cameras.insert_element(cad_pos="0,0", element=self.frame_camera_display_1, padx=5, pady=5, sticky="")
        self.frame_cameras.insert_element(cad_pos="0,1", element=self.frame_camera_display_2, padx=5, pady=5, sticky="")
        
        self.frame_buttons_messages=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,2), arr=np.array([["0,0","0,1"],["1,0","0,1"]])), fg_color='lightpink')
        self.frame_buttons_messages.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=self.frame_buttons_messages, text="Iniciar estimacion", size=16, weight="bold", command=self.button_start_estimation), padx=5, pady=5, sticky="")
        self.frame_buttons_messages.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=self.frame_buttons_messages, text="Cancelar estimacion", size=16, weight="bold", command=self.button_cancel_estimation, fg_color="red"), padx=5, pady=5, sticky="")
        self.frame_buttons_messages.insert_element(cad_pos="0,1", element=FrameHelp.create_label(master=self.frame_buttons_messages, text="", fg_color=utils.rgb_to_hex(color=(255,0,0)), width=100, height=100), padx=5, pady=5, sticky="")

        self.frame_estimation_3D_graphic=FrameEstimation3DGraphic(master=self, fg_color='lightpink')

        self.frame_estimation_configurations=FrameEstimationConfigurations(master=self, fg_color='lightpink')
        
        self.frame_extrinsic_matrices=FrameExtrinsicMatrices(master=self, fg_color='lightpink')

        self.insert_element(cad_pos="0,0", element=self.frame_cameras, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=self.frame_buttons_messages, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=self.frame_estimation_3D_graphic, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=self.frame_estimation_configurations, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=self.frame_extrinsic_matrices, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_label(master=self, text="", height=150, fg_color="gray80"), padx=5, pady=5, sticky="nsew")

        self.loop_verification_thread_estimation_calculation()

    def destroy(self):
        self.button_cancel_estimation()
        CreateScrollableFrame.destroy(self)

    def button_start_estimation(self):
        if self.thread_estimation_calculation is not None and self.thread_estimation_calculation.is_alive():
            # Primero cancelar antes de realizar una nueva estimacion
            tk.messagebox.showinfo(title="Calculo de matrices extrinsecas", message="Primero cancele la estimacion actual.")
        else:
            if self.thread_camera_1.calibration_information is not None and self.thread_camera_2.calibration_information is not None:
                Q1,Q2=self.frame_extrinsic_matrices.get_extrinsic_matrices(frame1=self.thread_camera_1.frame, K1=self.thread_camera_1.calibration_information.K, q1=self.thread_camera_1.calibration_information.q, frame2=self.thread_camera_2.frame, K2=self.thread_camera_2.calibration_information.K, q2=self.thread_camera_2.calibration_information.q)
                if Q1 is not None and Q2 is not None:
                    self.button_cancel_estimation()
                    configurations_dict=self.frame_estimation_configurations.get_configurations_dict()
                    self.thread_estimation_calculation=ThreadEstimationCalculation(thread_camera_1=self.thread_camera_1, thread_camera_2=self.thread_camera_2, Q1=Q1, Q2=Q2, configurations_dict=configurations_dict, estimations_dict_callback=self.estimations_dict_callback, daemon=True)
                    self.thread_estimation_calculation.start()
                else:
                    tk.messagebox.showinfo(title="Calculo de matrices extrinsecas", message="No se pudieron calcular las matrices extrinsecas.\n\nFavor de verificar que el patron sea visible en ambas camaras.")
            else:
                tk.messagebox.showinfo(title="Calculo de matrices extrinsecas", message="Alguna de las camaras no esta calibrada.\n\nFavor de verificar que ambas camaras esten calibradas.")

    def button_cancel_estimation(self):
        if self.thread_estimation_calculation is not None and self.thread_estimation_calculation.is_alive():
            self.thread_estimation_calculation.kill_thread()
        
    def loop_verification_thread_estimation_calculation(self):
        if self.thread_estimation_calculation is not None and not self.thread_estimation_calculation.is_alive():
            self.frame_estimation_3D_graphic.frame_3D_graphic.clear()
            self.frame_estimation_3D_graphic.frame_3D_graphic.canvas.draw()
            self.frame_buttons_messages.get_element(cad_pos="0,1").configure(fg_color=utils.rgb_to_hex(color=(255,0,0)))
            self.frame_camera_display_1.editable=False
            self.frame_camera_display_2.editable=False
            if self.verification_thread_estimation_calculation is not None:
                self.verification_thread_estimation_calculation(is_alive=False)
        elif self.thread_estimation_calculation is not None and self.thread_estimation_calculation.is_alive():
            self.frame_buttons_messages.get_element(cad_pos="0,1").configure(fg_color=utils.rgb_to_hex(color=(0,255,0)))
            self.frame_camera_display_1.editable=True
            self.frame_camera_display_2.editable=True
            if self.verification_thread_estimation_calculation is not None:
                self.verification_thread_estimation_calculation(is_alive=True)
        self.after(1000, self.loop_verification_thread_estimation_calculation)

    def estimations_dict_callback(self, estimations_dict):
        if not self.thread_estimation_calculation.event_kill_thread.is_set():
            self.frame_camera_display_1.update_label_camera(frame=estimations_dict['frames']['frame_1'])
            self.frame_camera_display_2.update_label_camera(frame=estimations_dict['frames']['frame_2'])

            conv_factor=0.1 # mm a cm
            configurations_dict=self.thread_estimation_calculation.configurations_dict
            for algorithm_name in estimations_dict['algorithms'].keys():
                if estimations_dict['algorithms'][algorithm_name]['is_double_estimate']:
                    vws_left=estimations_dict['algorithms'][algorithm_name]['left']['vws']
                    vws_right=estimations_dict['algorithms'][algorithm_name]['right']['vws']
                    d={"left": vws_left, "right": vws_right}
                    for key in d.keys():
                        vws=d[key]
                        if vws is not None:
                            if self.frame_estimation_3D_graphic.var_points.get():
                                self.frame_estimation_3D_graphic.frame_3D_graphic.plot_points(ps=conv_factor*CoordinateConverter.world_to_graphic(ps=vws[configurations_dict[algorithm_name][key]['enabled_points'],:]), rgb_color=configurations_dict[algorithm_name][key]['points_color'])
                            if self.frame_estimation_3D_graphic.var_connections.get():
                                self.frame_estimation_3D_graphic.frame_3D_graphic.plot_lines(ps=conv_factor*CoordinateConverter.world_to_graphic(ps=vws), connections=configurations_dict[algorithm_name][key]['connections'], rgb_color=configurations_dict[algorithm_name][key]['connections_color'])
                else:
                    vws=estimations_dict['algorithms'][algorithm_name]['vws']
                    if vws is not None:
                        if self.frame_estimation_3D_graphic.var_points.get():
                            self.frame_estimation_3D_graphic.frame_3D_graphic.plot_points(ps=conv_factor*CoordinateConverter.world_to_graphic(ps=vws[configurations_dict[algorithm_name]['enabled_points'],:]), rgb_color=configurations_dict[algorithm_name]['points_color'])
                        if self.frame_estimation_3D_graphic.var_connections.get():
                            self.frame_estimation_3D_graphic.frame_3D_graphic.plot_lines(ps=conv_factor*CoordinateConverter.world_to_graphic(ps=vws), connections=configurations_dict[algorithm_name]['connections'], rgb_color=configurations_dict[algorithm_name]['connections_color'])
            
            self.frame_estimation_3D_graphic.frame_3D_graphic.canvas.draw()
            self.frame_estimation_3D_graphic.frame_3D_graphic.clear()

            if self.estimations_results_callback is not None:
                self.estimations_results_callback(estimations_dict=estimations_dict)


