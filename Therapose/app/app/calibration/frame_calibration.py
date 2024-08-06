import customtkinter  as ctk
import tkinter as tk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame
from general.frame_help import FrameHelp
from general.frame_camera_display import FrameCameraDisplay
from general.thread_camera import ThreadCamera
from general.frame_progress_bar import FrameProgressBar

from calibration.frame_calibration_form import FrameCalibrationForm
from calibration.frame_take_photos_chessboard import FrameTakePhotosChessboard
from calibration.thread_refinement_process import ThreadRefinementProcess
from calibration.calibration_information import CalibrationInformation

class FrameCalibration(CreateScrollableFrame):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,1), arr=None), **kwargs)
        self.root=self.get_root()
        self.thread_refinement_process: ThreadRefinementProcess=None

        self.frame_cameras=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None), fg_color='lightpink')
        self.frame_cameras.insert_element(cad_pos="0,0", element=FrameCameraDisplay(master=self.frame_cameras, thread_camera=self.root.thread_camera_1, after_ms=10, scale_percent=50, show_calibration_information=True, editable=False, name="FrameCameraDisplayCalibration1"), padx=5, pady=5, sticky="")
        self.frame_cameras.insert_element(cad_pos="0,1", element=FrameCameraDisplay(master=self.frame_cameras, thread_camera=self.root.thread_camera_2, after_ms=10, scale_percent=50, show_calibration_information=True, editable=False, name="FrameCameraDisplayCalibration2"), padx=5, pady=5, sticky="")
        
        self.frame_cameras_buttons=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None), fg_color='lightpink')
        self.frame_cameras_buttons.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=self.frame_cameras_buttons, text="Calibrar camara 1", img_name="camera", compound=ctk.LEFT, command=lambda: self.camera_calibration(thread_camera=self.root.thread_camera_1)), padx=5, pady=5, sticky="")
        self.frame_cameras_buttons.insert_element(cad_pos="0,1", element=FrameHelp.create_button(master=self.frame_cameras_buttons, text="Calibrar camara 2", img_name="camera", compound=ctk.LEFT, command=lambda: self.camera_calibration(thread_camera=self.root.thread_camera_2)), padx=5, pady=5, sticky="")
        
        self.insert_element(cad_pos="0,0", element=self.frame_cameras, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=self.frame_cameras_buttons, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameProgressBar(master=self, title="Espere mientras se lleva a cabo el proceso de refinamiento de los parametros de la camara...", width=400, height=50), padx=5, pady=5, sticky="ew").hide_frame()
        self.insert_element(cad_pos="4,0", element=FrameCalibrationForm(master=self, fg_color='lightpink'), padx=5, pady=5, sticky="ew")
    
    def components_visibility(self, show):
        self.destroy_element(cad_pos="2,0")
        self.root.get_child(name="FrameMenu").show_frame() if show else self.root.get_child(name="FrameMenu").hide_frame()
        self.frame_cameras.get_element(cad_pos="0,0").calibration_information_visibility(show=show)
        self.frame_cameras.get_element(cad_pos="0,1").calibration_information_visibility(show=show)
        if not show: self.frame_cameras.get_element(cad_pos="0,0").close_window()
        if not show: self.frame_cameras.get_element(cad_pos="0,1").close_window()
        self.frame_cameras_buttons.show_frame() if show else self.frame_cameras_buttons.hide_frame()
        self.show_element(cad_pos="4,0") if show else self.hide_element(cad_pos="4,0")

    def camera_calibration(self, thread_camera: ThreadCamera):
        if thread_camera.camera_path != "":
            dict_calibration_options=self.get_element(cad_pos="4,0").get_fields()
            if dict_calibration_options is not None:
                if tk.messagebox.askyesnocancel(title="Calibracion de camara", message="¿Esta seguro de calibrar la camara '{}'?\n\nA continuacion se tomaran fotos del tablero de ajedrez para la calibracion.".format(thread_camera.camera_name)):
                    self.components_visibility(show=False)
                    self.insert_element(cad_pos="2,0", element=FrameTakePhotosChessboard(master=self, thread_camera=thread_camera, board_dimensions=dict_calibration_options["board_dimensions"], square_size=dict_calibration_options["square_size"], number_images=dict_calibration_options["number_images"], timer_time=dict_calibration_options["timer_time"], operation_callback=self.operation_callback, after_ms=10, fg_color='lightpink'), padx=5, pady=5, sticky="ew")
            else:
                tk.messagebox.showinfo(title="Calibracion de camara", message="Los campos del formulario de calibracion no son correctos.\n\nFavor de verifcarlos.")
        else:
            tk.messagebox.showinfo(title="Calibracion de camara", message="Debe seleccionar primero una camara.")

    def operation_callback(self, op, **kwargs):
        if op == "cancelled":
            self.components_visibility(show=True)
            tk.messagebox.showinfo(title="Calibracion de camara", message="Se ha cancelado el proceso de calibracion.")
        elif op == "completed":
            v3ps_list=kwargs['v3ps_list']
            l=len(v3ps_list)
            if l >= 3:
                self.thread_refinement_process=ThreadRefinementProcess(v3ps_real_list=v3ps_list, board_dimensions=self.get_element(cad_pos="2,0").board_dimensions, square_size=self.get_element(cad_pos="2,0").square_size)
                self.thread_refinement_process.start()
                self.get_element(cad_pos="3,0").start_progress_bar()
                self.wait_for_thread_refinement_process()
            else:
                tk.messagebox.showinfo(title="Calibracion de camara", message="No se han obtenido las suficientes imagenes buenas.\n\nPor lo menos se requieren 3 imagenes buenas.")
                self.components_visibility(show=True)

    def wait_for_thread_refinement_process(self):
        if self.thread_refinement_process.is_alive():
            self.after(1000, self.wait_for_thread_refinement_process)
        else:
            self.get_element(cad_pos="3,0").stop_progress_bar()
            if self.thread_refinement_process.is_ok:
                refinement_process=self.thread_refinement_process.refinement_process
                self.get_element(cad_pos="2,0").thread_camera.set_calibration_information(calibration_information=CalibrationInformation(board_dimensions=self.get_element(cad_pos="2,0").board_dimensions, square_size=self.get_element(cad_pos="2,0").square_size, number_images=len(self.thread_refinement_process.v3ps_real_list), timer_time=self.get_element(cad_pos="2,0").timer_time, K=refinement_process["K"], q=refinement_process["q"], Qs=refinement_process["Qs"], cost=refinement_process["cost"], jac_norm=refinement_process["jac_norm"]))
                tk.messagebox.showinfo(title="Calibracion de camara", message="Todo ha salido bien.\n\nLa calibracion de la camara fue exitosa.")
            else:
                tk.messagebox.showinfo(title="Calibracion de camara", message=self.thread_refinement_process.exception_message)
            self.components_visibility(show=True)
            
