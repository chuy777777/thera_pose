import customtkinter  as ctk
import cv2

import general.ui_images as ui_images
import general.utils as utils
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame
from general.frame_help import FrameHelp
from general.thread_camera import ThreadCamera
from general.frame_camera_display import FrameCameraDisplay

from calibration.calibration_functions import CalibrationFunctions
from calibration.patterns.frame_pattern_chessboard import FramePatternChessboard

class FrameTakePhotosChessboard(CreateFrame):
    def __init__(self, master, thread_camera: ThreadCamera, board_dimensions, square_size, number_images, timer_time, operation_callback, after_ms=10, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(11,1), arr=None), **kwargs) 
        self.thread_camera=thread_camera
        self.board_dimensions=board_dimensions
        self.square_size=square_size
        self.number_images=number_images
        self.timer_time=timer_time
        self.operation_callback=operation_callback
        self.after_ms=after_ms
        self.v3ps_list=[]
        self.var_time=ctk.StringVar(value="")

        self.frame_photos_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None), height=300, width=900)
        self.frame_information=CreateFrame(master=self.frame_photos_container, grid_frame=GridFrame(dim=(4,1), arr=None), height=200, width=300)
        self.frame_information.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_information, text="Proceso de captura de imagenes", size=14, weight="bold"), padx=5, pady=5, sticky="ew")
        self.frame_information.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self.frame_information, text="'{}'".format(self.thread_camera.camera_name), weight="bold"), padx=5, pady=5, sticky="ew")
        self.frame_information.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self.frame_information, text="", textvariable=self.var_time, weight="bold"), padx=5, pady=5, sticky="ew")
        self.frame_information.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self.frame_information, text="Cancelar", width=125, height=35, command=self.button_cancel, fg_color="#FF3030", hover_color="#B22222"), padx=5, pady=5, sticky="ew")
        self.frame_information.enable_fixed_size()
        self.frame_photos_list=CreateScrollableFrame(master=self.frame_photos_container, grid_frame=GridFrame(dim=(1, self.number_images), arr=None), height=400, width=600, orientation="horizontal")
        self.frame_photos_container.insert_element(cad_pos="0,0", element=self.frame_information, padx=5, pady=5, sticky="ew")
        self.frame_photos_container.insert_element(cad_pos="0,1", element=self.frame_photos_list, padx=5, pady=5, sticky="ew")
        self.frame_photos_container.enable_fixed_size()

        self.insert_element(cad_pos="0,0", element=self.frame_photos_container, padx=5, pady=5, sticky="ew")

        self.count(index=0, t=self.timer_time)

    def count(self, index, t):
        if index < self.number_images:
            if t > 0:
                conv_factor=0.001   # ms a s 
                t-=self.after_ms*conv_factor
                if t < 0:
                    t=0
                self.var_time.set(value="Foto '{}' de '{}': '{}'".format(index+1, self.number_images, round(t, 2)))
            else:
                index+=1
                t=self.timer_time
                self.capture_image(index=index)
            self.after(self.after_ms, lambda: self.count(index=index, t=t))
        else:
            self.var_time.set(value="")
            self.frame_information.get_element(cad_pos="3,0").configure(state=ctk.DISABLED)
            self.operation_callback(op="completed", v3ps_list=self.v3ps_list)

    def capture_image(self, index):
        frame=self.thread_camera.frame
        if frame is not None:
            v3ps=CalibrationFunctions.get_v3ps_from_chessboard_image(frame=frame, board_dimensions=self.board_dimensions, fast=False)
            if v3ps is not None:
                new_frame=CalibrationFunctions.draw_points_on_frame(frame=frame, v3ps=v3ps)
                new_frame=cv2.putText(new_frame, '{}'.format(index), (10,60), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                new_frame=utils.resize_image(scale_percent=50, img=new_frame)
                img=utils.frame_to_img(frame=new_frame)
                self.frame_photos_list.insert_element(cad_pos="0,{}".format(self.number_images - index), element=ctk.CTkLabel(master=self.frame_photos_list, text="", image=img), padx=5, pady=5)
                self.v3ps_list.append(v3ps)
            else:
                new_frame=frame.copy()
                new_frame=cv2.putText(new_frame, '{}'.format(index), (10,60), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                new_frame=utils.resize_image(scale_percent=50, img=new_frame)
                new_frame[:,:,0]=255
                img=utils.frame_to_img(frame=new_frame)
                self.frame_photos_list.insert_element(cad_pos="0,{}".format(self.number_images - index), element=ctk.CTkLabel(master=self.frame_photos_list, text="", image=img), padx=5, pady=5)
        else:
            self.frame_photos_list.insert_element(cad_pos="0,{}".format(self.number_images - index), element=FrameHelp.create_label(master=self.frame_photos_list, text="", img_name="no_image_available", img_width=250, img_height=250), padx=5, pady=5)

    def button_cancel(self):
        self.operation_callback(op="cancelled")
     
