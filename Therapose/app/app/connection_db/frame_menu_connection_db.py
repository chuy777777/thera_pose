import tkinter as tk
import customtkinter  as ctk

import general.ui_images as ui_images
from general.create_frame import CreateFrame
from general.grid_frame import GridFrame
from general.frame_help import FrameHelp
from general.frame_login import FrameLogin
from general.application_message_types import MessageTypesConnectionDB
from general.create_window import CreateWindow

from connection_db.connection_db import ConnectionDB

from database_application.frame_database_application import FrameDatabaseApplication

class FrameMenuConnectionDB(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs) 
        self.var_connection_type=ctk.StringVar(value="Local")
        self.window_database_application=None
        self.primary_connection_db: ConnectionDB=None

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_option_menu(master=self, weight="bold", size=14, variable=self.var_connection_type, values=["Local", "Cloud"]), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameLogin(master=self, button_login=self.button_login, button_logout=self.button_logout), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=self, text="Abrir aplicacion", size=14, weight="bold", fg_color="coral", img_name="open", img_width=30, img_height=30, command=self.create_window), padx=5, pady=5, sticky="")

        self.hide_element(cad_pos="2,0")
        
    def destroy(self):
        self.close_db_connection()
        CreateFrame.destroy(self)
    
    def button_login(self, username, password):
        connection_type='local' if self.var_connection_type.get() == 'Local' else 'cloud'
        self.primary_connection_db=ConnectionDB.create_connection_db(username=username, password=password, connection_type=connection_type)
        self.primary_connection_db.operation(op="create_connection")
        if self.primary_connection_db.message_notification_connection_db.message_type == MessageTypesConnectionDB.CREATE_CONNECTION_OK:
            self.primary_connection_db.operation(op="build_database_classes")
            if self.primary_connection_db.message_notification_connection_db.message_type == MessageTypesConnectionDB.BUILD_DATABASE_CLASSES_OK:
                self.create_window()
                self.get_element(cad_pos="1,0").alternate()
                self.show_element(cad_pos="2,0")
            else:
                tk.messagebox.showinfo(title=self.primary_connection_db.message_notification_connection_db.message, message=self.primary_connection_db.message_notification_connection_db.specific_message)
        else:
            tk.messagebox.showinfo(title=self.primary_connection_db.message_notification_connection_db.message, message=self.primary_connection_db.message_notification_connection_db.specific_message)

    def button_logout(self):
        self.get_element(cad_pos="1,0").alternate()
        self.close_db_connection()
        self.hide_element(cad_pos="2,0")
        if self.window_database_application is not None:
            self.window_database_application.close_window()

    def close_db_connection(self):
        if self.primary_connection_db is not None:
            # Cerramos la conexion con la base de datos
            self.primary_connection_db.operation(op="close")
            self.primary_connection_db=None

    def create_window(self):
        if self.primary_connection_db is not None and self.window_database_application is None:
            root=self.get_root()
            self.window_database_application=CreateWindow(window_title="Aplicacion de base de datos", window_geometry=(1366, 768), on_closing_callback=self.close_window_callback, scrollable=False, padx=5, pady=5, sticky="nsew")
            self.window_database_application.frame_root.insert_element(cad_pos="0,0", element=FrameDatabaseApplication(master=self.window_database_application.frame_root, primary_connection_db=self.primary_connection_db, thread_camera_1=root.thread_camera_1, thread_camera_2=root.thread_camera_2, is_root=True, name="FrameDatabaseApplication"), padx=5, pady=5, sticky="nsew")
        else:
            tk.messagebox.showinfo(title="Aplicacion de base de datos", message="La aplicacion de base de datos ya esta iniciada.")
        
    def close_window_callback(self):
        self.window_database_application=None