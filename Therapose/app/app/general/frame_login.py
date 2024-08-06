import numpy as np
import customtkinter  as ctk
import tkinter as tk

import general.ui_images as ui_images
from general.frame_entry_validation import FrameEntryValidation, GroupClassValidation, TextValidation
from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.build_component import BuildComponent
from general.frame_help import FrameHelp

class FrameLogin(CreateFrame, BuildComponent):
    def __init__(self, master, button_login, button_logout, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        BuildComponent.__init__(self)
        self.button_login=button_login
        self.button_logout=button_logout
        self.var_username=ctk.StringVar(value="")
        self.var_password=ctk.StringVar(value="")
        
        self.build_component()

    def build_component(self):
        self.destroy_all()
        self.frame_success=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,1), arr=np.array([["0,0"],["0,0"],["0,0"],["3,0"]])))
        self.frame_success.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_success, text="La sesion se ha iniciado"), padx=5, pady=5, sticky="")
        self.frame_success.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self.frame_success, text="Cerrar sesion", width=100, command=self.logout, img_name="logout", compound=ctk.LEFT, weight="bold", fg_color="coral"), padx=5, pady=5, sticky="")
        
        self.frame_login=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,1), arr=None))
        self.frame_login.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_login, text="Inicio de sesion"), padx=5, pady=5, sticky="")
        self.frame_login.insert_element(cad_pos="1,0", element=FrameEntryValidation(master=self.frame_login, class_validation=TextValidation(var_entry=self.var_username, max_length=20), text_title="Usuario", text_show=""), padx=5, pady=5, sticky="")
        self.frame_login.insert_element(cad_pos="2,0", element=FrameEntryValidation(master=self.frame_login, class_validation=TextValidation(var_entry=self.var_password, max_length=20), text_title="Contraseña", text_show="*"), padx=5, pady=5, sticky="")
        self.frame_login.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self.frame_login, text="Iniciar sesion", width=100, command=self.login, weight="bold", fg_color="coral"), padx=5, pady=5, sticky="")
        
        self.group_class_validation=GroupClassValidation(list_class_validation=[
            self.frame_login.get_element(cad_pos="1,0").class_validation, 
            self.frame_login.get_element(cad_pos="2,0").class_validation
        ])

        self.insert_element(cad_pos="0,0", element=self.frame_success, padx=5, pady=5, sticky="").hide_frame()
        self.insert_element(cad_pos="1,0", element=self.frame_login, padx=5, pady=5, sticky="")

    def clear(self):
        self.var_username.set(value="")
        self.var_password.set(value="")

    def alternate(self):
        if self.frame_success.is_visible:
            self.clear()
            self.frame_success.hide_frame()
            self.frame_login.show_frame()
        else:
            self.frame_success.show_frame()
            self.frame_login.hide_frame()

    def login(self):
        if self.group_class_validation.is_valid():
            self.button_login(username=self.var_username.get(), password=self.var_password.get())

    def logout(self):
        if tk.messagebox.askyesnocancel(title="Cerrar sesion", message="¿Desea cerrar la sesion?"):
            self.button_logout()

    


