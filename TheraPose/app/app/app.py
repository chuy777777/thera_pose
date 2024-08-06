import numpy as np
import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.thread_camera import ThreadCamera

from menu.frame_menu import FrameMenu
from home.frame_home import FrameHome
from calibration.frame_calibration import FrameCalibration
from estimation.frame_estimation import FrameEstimation
from game.frame_game import FrameGame

class FramePrincipalApplication(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,10), arr=np.array([["0,0","0,1","0,1","0,1","0,1","0,1","0,1","0,1","0,1","0,1"]])), **kwargs)
        self.thread_camera_1=ThreadCamera(root_name="camera_1", camera_name="", camera_path="", camera_changes_callback=self.camera_changes_callback, daemon=True)
        self.thread_camera_2=ThreadCamera(root_name="camera_2", camera_name="", camera_path="", camera_changes_callback=self.camera_changes_callback, daemon=True)
        self.thread_camera_1.start()
        self.thread_camera_2.start()
        
        self.insert_element(cad_pos="0,0", element=FrameMenu(master=self, name="FrameMenu"), padx=5, pady=5, sticky="nsew")

    def destroy(self):
        self.thread_camera_1.kill_thread()
        self.thread_camera_2.kill_thread()
        CreateFrame.destroy(self)

    def select_page(self, page_name):
        self.destroy_element(cad_pos="0,1")

        if page_name == "home":
            self.insert_element(cad_pos="0,1", element=FrameHome(master=self, name="FrameHome", fg_color='lightpink'), padx=5, pady=5, sticky="nsew")
        elif page_name == "calibration":
            self.insert_element(cad_pos="0,1", element=FrameCalibration(master=self, name="FrameCalibration"), padx=5, pady=5, sticky="nsew")
        elif page_name == "estimation":
            self.insert_element(cad_pos="0,1", element=FrameEstimation(master=self, name="FrameEstimation"), padx=5, pady=5, sticky="nsew")
        elif page_name == "game":
            self.insert_element(cad_pos="0,1", element=FrameGame(master=self, name="FrameGame"), padx=5, pady=5, sticky="nsew")
            
    def camera_changes_callback(self, root_name, camera_name):
        frame_camera_display_calibration_1=self.get_child(name="FrameCameraDisplayCalibration1")
        frame_camera_display_calibration_2=self.get_child(name="FrameCameraDisplayCalibration2")
        frame_estimation=self.get_child(name="FrameEstimation")
        if frame_camera_display_calibration_1 is not None:
            if frame_camera_display_calibration_1.window is not None and root_name == "camera_1":
                frame_camera_display_calibration_1.close_window()
            frame_camera_display_calibration_1.camera_changes()
        if frame_camera_display_calibration_2 is not None:
            if frame_camera_display_calibration_2.window is not None and root_name == "camera_2":
                frame_camera_display_calibration_2.close_window()
            frame_camera_display_calibration_2.camera_changes()
        if frame_estimation is not None and frame_estimation.thread_estimation_calculation is not None and frame_estimation.thread_estimation_calculation.is_alive():
            frame_estimation.thread_estimation_calculation.kill_thread()

class App(ctk.CTk):
    def __init__(self, **kwargs):
        ctk.CTk.__init__(self)
        self.frame_principal_application=None
        
        # Configuramos nuestra aplicacion
        self.geometry("1366x768")
        self.title("Aplicacion principal")

        # Configuramos el sistema de cuadricula
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Creamos un frame root
        self.frame_root=CreateFrame(master=self, grid_frame=GridFrame())

        # Colocamos el frame root en la cuadricula
        self.frame_root.grid(row=0, column=0, padx=5, pady=5, sticky="nsew") # Al agregar sticky='nsew' el frame pasa de widthxheight a abarcar todo el espacio disponible

        # Creamos elementos y los insertamos a los frames
        self.frame_principal_application=FramePrincipalApplication(master=self.frame_root, is_root=True, name="FramePrincipalApplication", fg_color="gray50")

        self.frame_root.insert_element(cad_pos="0,0", element=self.frame_principal_application, padx=5, pady=5, sticky="nsew")

        # Configuramos el cerrado de la ventana
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def destroy(self):
        """
        La razon de hacer esto es por este problema:
            Error after closing the window with Matplotlib figure

        No es nada malo pero con esto evitamos que se bloquee la terminal.
        """
        ctk.CTk.quit(self)
        ctk.CTk.destroy(self)

if __name__ == "__main__":

    # Configuramos e iniciamos la aplicacion
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("green")
    app=App()
    app.mainloop()

    # # Se ejecutan estas lineas una vez que se cierra la aplicacion
    # print('After mainlooop')
    # global_thread_camera_1.kill_thread()
    # global_thread_camera_1.join()
    # global_thread_camera_2.kill_thread()
    # global_thread_camera_2.join()
    # print("End all threads")





