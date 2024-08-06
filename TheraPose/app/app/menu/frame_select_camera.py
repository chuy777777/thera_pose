import numpy as np
import customtkinter  as ctk
import subprocess
import re

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp
from general.device import Device

class FrameSelectCamera(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,1), arr=np.array([["0,0"],["1,0"],["2,0"],["3,0"],["4,0"]])), **kwargs) 
        self.root=self.get_root()
        self.list_devices=[]
        self.none_camera="----- None -----"

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=self, text="Recargar", command=self.reload, width=100, weight="bold", fg_color="coral"), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, text="Camara 1"), padx=5, pady=5)
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_option_menu(master=self, variable=ctk.StringVar(value=self.root.thread_camera_1.camera_name), command=lambda camera_name: self.change_camera(element=self.get_element(cad_pos="2,0"), camera_name=camera_name, primary_thread_camera=self.root.thread_camera_1, secondary_thread_camera=self.root.thread_camera_2)), padx=5, pady=5)
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_label(master=self, text="Camara 2"), padx=5, pady=5)
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_option_menu(master=self, variable=ctk.StringVar(value=self.root.thread_camera_2.camera_name), command=lambda camera_name: self.change_camera(element=self.get_element(cad_pos="4,0"), camera_name=camera_name, primary_thread_camera=self.root.thread_camera_2, secondary_thread_camera=self.root.thread_camera_1)), padx=5, pady=5)

        self.reload()

        # Configuracion inicial (solo en modo de prueba)
        # camera_name_1="HX-USB Camera: HX-USB Camera"
        # camera_name_2="USB PHY 2.0: USB CAMERA"
        # self.get_element(cad_pos="2,0").configure(variable=ctk.StringVar(value=camera_name_1))
        # self.get_element(cad_pos="4,0").configure(variable=ctk.StringVar(value=camera_name_2))
        # self.change_camera(element=None, camera_name=camera_name_1, primary_thread_camera=self.root.thread_camera_1, secondary_thread_camera=self.root.thread_camera_2)
        # self.change_camera(element=None, camera_name=camera_name_2, primary_thread_camera=self.root.thread_camera_2, secondary_thread_camera=self.root.thread_camera_1)

    def reload(self):
        self.list_devices=self.get_devices()
        values=[dev.name for dev in self.list_devices]
        values.insert(0, self.none_camera)
        if self.root.thread_camera_1.camera_name not in values: 
            if self.get_element(cad_pos="2,0").get() == self.root.thread_camera_1.camera_name: self.get_element(cad_pos="2,0").configure(variable=ctk.StringVar(value=self.none_camera))
            self.root.thread_camera_1.init_cap(camera_name=self.none_camera, camera_path="")
        if self.root.thread_camera_2.camera_name not in values: 
            if self.get_element(cad_pos="4,0").get() == self.root.thread_camera_2.camera_name: self.get_element(cad_pos="4,0").configure(variable=ctk.StringVar(value=self.none_camera))
            self.root.thread_camera_2.init_cap(camera_name=self.none_camera, camera_path="")
        self.get_element(cad_pos="2,0").configure(values=values)
        self.get_element(cad_pos="4,0").configure(values=values)

    def get_devices(self):
        list_devices=[]
        res=subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True)
        res_split=res.stdout.split("\n\n")[0:-1]
        pattern = "video[0-9]{1}"
        for r in res_split:
            r_split=r.split("\n\t")
            if len(re.findall(pattern, r_split[1]))>0:
                dev_name=r_split[0].split("(")[0].strip() 
                dev_path=r_split[1]
                list_devices.append(Device(name=dev_name, path=dev_path))
        return list_devices
    
    def change_camera(self, element, camera_name, primary_thread_camera, secondary_thread_camera):
        if camera_name == self.none_camera: 
            primary_thread_camera.init_cap(camera_name=self.none_camera, camera_path="")
        elif camera_name != secondary_thread_camera.camera_name and camera_name != primary_thread_camera.camera_name:
            for dev in self.list_devices:
                if dev.name==camera_name:
                    primary_thread_camera.init_cap(camera_name=camera_name, camera_path=dev.path)
                    break
        else:
            element.configure(variable=ctk.StringVar(value=primary_thread_camera.camera_name))
