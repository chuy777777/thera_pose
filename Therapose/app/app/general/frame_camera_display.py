import numpy as np
import customtkinter  as ctk
import cv2

import general.ui_images as ui_images
import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.thread_camera import ThreadCamera
from general.create_window import CreateWindow
from general.frame_3D_graphic import Frame3DGraphic

from calibration.calibration_functions import CalibrationFunctions

class FrameCalibrationInformation(CreateFrame):
    def __init__(self, master, thread_camera: ThreadCamera, after_ms, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs) 
        self.thread_camera=thread_camera
        self.after_ms=after_ms

        self.frame_3D_graphic=Frame3DGraphic(master=self)

        self.frame_camera_display=FrameCameraDisplay(master=self, thread_camera=self.thread_camera, after_ms=self.after_ms, scale_percent=100, show_calibration_information=False, editable=True)

        self.insert_element(cad_pos="0,0", element=self.frame_3D_graphic, padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=self.frame_camera_display, padx=5, pady=5)

        self.draw_calibration_planes()
        self.draw_calibration_information()
        self.draw_3D_cube()

    def draw_calibration_planes(self):
        self.frame_3D_graphic.clear()
        self.frame_3D_graphic.graphic_configuration(title="")
        self.frame_3D_graphic.canvas.draw()
        calibration_information=self.thread_camera.calibration_information
        if calibration_information is not None:
            # Trabajaremos con cm (1 mm = 0.1 cm)
            conv_factor=0.1
            axis_length=20 
            # Declaramos matrices
            T1=np.eye(3,3)
            T21=np.concatenate([T1[:,[0]], -T1[:,[2]], T1[:,[1]]],axis=1)
            t21=np.zeros((3,1))*conv_factor
            # Declaramos matriz de transformacion homogenea
            H21=np.concatenate([np.concatenate([T21, t21], axis=1), np.array([[0,0,0,1]])], axis=0)
            # Declaramos matrices auxiliares
            points_coor_sys=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
            points_coor_sys[0:3,:]*=axis_length
            plane_verts=np.concatenate([np.concatenate([points_coor_sys[0:3,[0]] + -points_coor_sys[0:3,[1]], -points_coor_sys[0:3,[0]] + -points_coor_sys[0:3,[1]], -points_coor_sys[0:3,[0]] + points_coor_sys[0:3,[1]], points_coor_sys[0:3,[0]] + points_coor_sys[0:3,[1]]], axis=1), np.ones((1,4))], axis=0)
            # Configuramos el grafico
            self.frame_3D_graphic.graphic_configuration(title="Planos del tablero de ajedrez (unidades: cm)\nSe utilizaron {} planos".format(calibration_information.number_images), xlabel="X", ylabel="Y", zlabel="Z", xlim=[-50,50], ylim=[-0,100], zlim=[-10,50])
            # Inicializamos el grafico
            self.frame_3D_graphic.plot_coordinate_system(t=t21, T=(T21)*axis_length)
            vhs=H21@plane_verts
            self.frame_3D_graphic.plot_plane(xs=vhs[0,:], ys=vhs[1,:], zs=vhs[2,:], facecolors='r', alpha=0.2)
            # Dibujamos los planos
            for i in range(calibration_information.number_images):
                Q=calibration_information.Qs[i]
                T32=Q[:,[0,1,2]]                                # [iw jw kw]    (los vectores forman una base ortonormal)
                t32=Q[:,[3]]*conv_factor                        # tw            (convertimos de mm a cm)
                H32=np.concatenate([np.concatenate([T32, t32], axis=1), np.array([[0,0,0,1]])], axis=0)
                self.frame_3D_graphic.plot_coordinate_system(t=t21 + T21@t32, T=(T21@T32)*axis_length)
                vhs=H21@H32@plane_verts
                self.frame_3D_graphic.plot_plane(xs=vhs[0,:], ys=vhs[1,:], zs=vhs[2,:], alpha=0.2)
            self.frame_3D_graphic.canvas.draw()

    def draw_calibration_information(self):
        self.destroy_element(cad_pos="2,0")
        calibration_information=self.thread_camera.calibration_information
        if calibration_information is not None:
            self.frame_calibration_process_results=CreateFrame(master=self, grid_frame=GridFrame(dim=(10,1), arr=None))
            self.frame_calibration_process_results.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_calibration_process_results, text="Configuraciones de calibracion", size=14, weight="bold", fg_color="coral"), padx=5, pady=5)
            self.frame_calibration_process_results.insert_element(cad_pos="1,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=250, height=100, title="Dimensiones del tablero (horizontal,vertical)", text="({}, {})".format(calibration_information.board_dimensions[0], calibration_information.board_dimensions[1])), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="2,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=100, title="Tamano de los cuadrados", text="{} mm".format(calibration_information.square_size)), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="3,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=100, title="Numero de imagenes", text="{}".format(calibration_information.number_images)), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="4,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=100, title="Tiempo entre capturas", text="{} segundo(s)".format(calibration_information.timer_time)), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="5,0", element=FrameHelp.create_label(master=self.frame_calibration_process_results, text="Resultados del proceso de calibracion de la camara", size=14, weight="bold", fg_color="coral"), padx=5, pady=5)
            self.frame_calibration_process_results.insert_element(cad_pos="6,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=200, title="Matriz intrinseca", text=np.array2string(calibration_information.K, separator=', ', formatter={'float_kind':lambda x: "%.2f" % x})), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="7,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=200, title="Parametros de distorsion (k1,k2,k3,p1,p2)", text=np.array2string(calibration_information.q, separator=', ', formatter={'float_kind':lambda x: "%.6f" % x})), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="8,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=100, title="Costo de la funcion objetivo", text=str(calibration_information.cost)), padx=5, pady=5, sticky="ew")
            self.frame_calibration_process_results.insert_element(cad_pos="9,0", element=FrameHelp.create_table_sub_item(master=self.frame_calibration_process_results, width=200, height=100, title="Norma del jacobiano (debe ser 0)", text=str(calibration_information.jac_norm)), padx=5, pady=5, sticky="ew")
            self.insert_element(cad_pos="2,0", element=self.frame_calibration_process_results, padx=5, pady=5)

    def draw_3D_cube(self):
        calibration_information=self.thread_camera.calibration_information
        if calibration_information is not None:
            frame=self.frame_camera_display.frame
            new_frame=CalibrationFunctions.draw_3D_cube_on_chessboard(frame=frame, K=calibration_information.K, q=calibration_information.q, board_dimensions=calibration_information.board_dimensions, square_size=calibration_information.square_size)
            self.frame_camera_display.update_label_camera(frame=frame if new_frame is None else new_frame)
        self.after(self.after_ms, self.draw_3D_cube)

class FrameCameraDisplay(CreateFrame):
    def __init__(self, master, thread_camera: ThreadCamera, after_ms=10, scale_percent=100, show_calibration_information=True, editable=False, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,1), arr=np.array([["0,0"],["1,0"],["1,0"],["1,0"],["1,0"]])), **kwargs) 
        self.thread_camera=thread_camera
        self.after_ms=after_ms
        self.scale_percent=scale_percent
        self.editable=editable
        self.frame=None
        self.window=None
        
        self.frame_calibration_information=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,1), arr=None))
        self.frame_calibration_information.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_calibration_information), padx=5, pady=5, sticky="ew")
        self.frame_calibration_information.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self.frame_calibration_information), padx=5, pady=5, sticky="ew")
        self.frame_calibration_information.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=self.frame_calibration_information, text="Informacion de la calibracion", command=self.create_window), padx=5, pady=1)
        
        self.calibration_information_visibility(show=show_calibration_information)

        self.insert_element(cad_pos="0,0", element=self.frame_calibration_information, padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, text=""), padx=5, pady=5)

        self.camera_changes()
        self.show_camera_view()

    def calibration_information_visibility(self, show):
        self.frame_calibration_information.show_element(cad_pos="2,0") if show else self.frame_calibration_information.hide_element(cad_pos="2,0")

    def create_window(self):
        if self.window is None:
            self.window=CreateWindow(window_title="Informacion de la calibracion", window_geometry=(800, 600), on_closing_callback=self.close_window_callback, scrollable=True, padx=20, pady=20, sticky="nsew")
            self.window.frame_root.insert_element(cad_pos="0,0", element=FrameCalibrationInformation(master=self.window.frame_root, thread_camera=self.thread_camera, after_ms=self.after_ms, fg_color='lightpink'), padx=20, pady=20, sticky="nsew")
    
    def close_window(self):
        if self.window is not None:
            self.window.destroy()
            self.window=None

    def close_window_callback(self):
        self.window=None

    def camera_changes(self):            
        self.frame_calibration_information.get_element(cad_pos="0,0").configure(text=self.thread_camera.camera_name)
        self.frame_calibration_information.get_element(cad_pos="1,0").configure(text="Camara calibrada" if self.thread_camera.calibration_information is not None else "Camara no calibrada")
        self.frame_calibration_information.get_element(cad_pos="2,0").configure(state="normal" if self.thread_camera.calibration_information is not None else "disabled")

    def not_camera(self):
        image=ui_images.get_image(name="no_image_available", width=250, height=250)
        self.get_element(cad_pos="1,0").configure(image=image)
        self.get_element(cad_pos="1,0").image=image

    def update_label_camera(self, frame):
        if frame is not None:
            if self.scale_percent != 100:
                frame=utils.resize_image(scale_percent=self.scale_percent, img=frame)
                self.frame=frame
            img=utils.frame_to_img(frame)
            self.get_element(cad_pos="1,0").configure(image=img)
            self.get_element(cad_pos="1,0").image=img
        else:
            self.frame=None

    def show_camera_view(self):
        self.frame=self.thread_camera.frame 
        if self.frame is None:
            self.not_camera()
        elif not self.editable:
            self.update_label_camera(frame=self.frame)
        self.after(self.after_ms, self.show_camera_view)  