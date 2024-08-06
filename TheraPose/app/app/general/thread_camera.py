import cv2
import time
from threading import Thread

from calibration.calibration_information import CalibrationInformation
from general.template_thread import TemplateThread

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

class ThreadCameraNotifier:
    def __init__(self):
        self.camera_callbacks=[]

    def add_camera_callback(self, camera_callback):
        self.camera_callbacks.append(camera_callback)
    
    def call_camera_callbacks(self):
        for camera_callback in self.camera_callbacks:
            camera_callback()

    def remove_camera_callback(self, camera_callback):
        try:
            self.camera_callbacks.remove(camera_callback)
            print("Removed function callback")
        except ValueError:
            print("Function is not in list")

    def clear_camera_callbacks(self):
        self.camera_callbacks=[]

class ThreadCamera(Thread, TemplateThread, ThreadCameraNotifier):
    def __init__(self, root_name: Literal["camera_1", "camera_2"], camera_name, camera_path="/dev/video0", camera_changes_callback=None, **kwargs):
        Thread.__init__(self, **kwargs)
        TemplateThread.__init__(self)
        ThreadCameraNotifier.__init__(self)
        self.root_name=root_name
        self.camera_name=camera_name
        self.camera_path=camera_path
        self.camera_changes_callback=camera_changes_callback
        self.cap=None
        self.frame=None
        self.calibration_information: CalibrationInformation=None

    def run(self):
        self.init_cap(camera_name=self.camera_name, camera_path=self.camera_path)
        self.view_camera()       
        print("End ThreadCamera from '{}'".format(self.camera_name))    

    def init_cap(self, camera_name, camera_path):
        self.release_cap()
        self.camera_name=camera_name
        self.camera_path=camera_path
        self.load_calibration_information()
        if self.camera_path != "":
            self.cap=cv2.VideoCapture(self.camera_path)
        else:
            self.cap=cv2.VideoCapture()
        
    def release_cap(self):
        if self.cap is not None:
            self.cap.release()

    def view_camera(self):
        while not self.event_kill_thread.is_set():
            try:
                if self.cap is not None:
                    if self.cap.isOpened():
                        ret, frame=self.cap.read()
                        if ret:
                            self.frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            """
                            Una vez que esta funcionando, si la camara se desconecta entra aqui.
                            Si se vuelve a conectar por USB no se reconecta por si sola.
                            """
                            self.frame=None
                    else:
                        """
                        Si la camara no esta conectada por USB desde un inicio entra aqui.
                        Si se vuelve a conectar por USB no se reconecta por si sola.
                        """
                        self.frame=None
                else:
                    self.frame=None
            except cv2.error as e:
                print("(cv2.error): {}".format(e))

            time.sleep(0.001)
        self.release_cap()
    
    def load_calibration_information(self):
        self.calibration_information=CalibrationInformation.load(camera_name=self.camera_name)
        if self.camera_changes_callback is not None: 
            self.camera_changes_callback(root_name=self.root_name, camera_name=self.camera_name)

    def save_calibration_information(self):
        if self.calibration_information is not None:
            self.calibration_information.save(camera_name=self.camera_name)

    def set_calibration_information(self, calibration_information: CalibrationInformation):
        self.calibration_information=calibration_information
        self.save_calibration_information()
        if self.camera_changes_callback is not None: 
            self.camera_changes_callback(root_name=self.root_name, camera_name=self.camera_name)
