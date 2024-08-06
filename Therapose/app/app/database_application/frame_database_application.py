import numpy as np
import customtkinter  as ctk
from CTkListbox import CTkListbox
import tkinter as tk
import re
from bson import ObjectId 
import copy
from tkcalendar import Calendar, DateEntry
from datetime import datetime
import os
import shutil
import cv2

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame
import general.ui_images as ui_images
import general.utils as utils
from general.application_notifications import MessageNotificationConnectionDB
from general.application_message_types import MessageTypesConnectionDB
from general.build_component import BuildComponent
from general.frame_help import FrameHelp

import connection_db.utils_database as utils_database
from connection_db.build_db_component import BuildDBComponent
from connection_db.connection_db import ConnectionDB
from connection_db.database_classes.patient import Patient
from connection_db.database_classes.disease import Disease
from connection_db.database_classes.pathology import Pathology

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

"""
Atajos teclado:
    - Contraer todo el codigo:
        Ctrl + K, Ctrl + 0[cero]
    - Expandir todo el codigo:
        Ctrl + K, Ctrl + J

        Ctrl + K, Ctrl + [ Contraer el codigo en mi posicion del cursor
        Ctrl + K, Ctrl + ] Expandir el codigo en mi posicion del cursor

Aqui mismo implementar los metodos para guardar en la base de datos local/nube 
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class TemplateFrameClasses():
    def __init__(self, **kwargs):
        pass

    # Este metodo se debe sobreescribir
    def clear_all(self):
        pass

    # Este metodo se debe sobreescribir
    def set_registry(self, registry):
        pass

    # Este metodo se debe sobreescribir
    def get_registry(self):
        return None

class FrameCalendarDate(CreateFrame):
    def __init__(self, master, calendar_text_size=14, calendar_visualization_text_size=14, calendar_width=500, calendar_height=500, calendar_callback=None, initial_date=utils_database.get_date_today().split(" ")[0], cal_event_dates=None, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs) 
        self.cal_event_dates=[]
        self.cal_event_ids=[]
        self.calendar_callback=calendar_callback
        self.var_date_visualization=ctk.StringVar(value=initial_date) 
        self.var_calendar=ctk.StringVar(value=initial_date) 
        self.var_calendar.trace_add("write", self.calendar_trace_callback)
        self.font_calendar=ctk.CTkFont(family="Modern", size=calendar_text_size, weight="bold")

        self.frame_calendar_container=CreateFrame(master=self, grid_frame=GridFrame(), fg_color="lightcoral", width=calendar_width, height=calendar_height)
        self.frame_calendar_container.enable_fixed_size()
        self.frame_calendar=Calendar(master=self.frame_calendar_container, textvariable=self.var_calendar, selectmode="day", year=int(initial_date.split("-")[0]), month=int(initial_date.split("-")[1]), day=int(initial_date.split("-")[2]), date_pattern="y-mm-dd", font=self.font_calendar)
        self.frame_calendar_container.insert_element(cad_pos="0,0", element=self.frame_calendar, padx=5, pady=5, sticky="nsew")

        self.frame_visualization=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,1), arr=None))
        self.frame_visualization.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_visualization, textvariable=self.var_date_visualization, text="", weight="bold", size=calendar_visualization_text_size), padx=5, pady=5, sticky="w")

        self.insert_element(cad_pos="0,0", element=self.frame_calendar_container, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=self.frame_visualization, padx=5, pady=5, sticky="")

        if cal_event_dates is not None:
            self.add_cal_event_dates(cal_event_dates=cal_event_dates)

        # Devuelve un diccionario muy completo con todos los eventos actuales del calendario
        # self.frame_calendar.calevents

    def clear_cal_event_dates(self):
        for cal_event_id in self.cal_event_ids:
            self.frame_calendar.calevent_remove(cal_event_id)
        self.cal_event_dates=[]
        self.cal_event_ids=[]
        
    def add_cal_event_dates(self, cal_event_dates, text=None):
        fmt="%Y-%m-%d %H:%M:%S"
        self.cal_event_dates.extend(cal_event_dates)
        for i in range(len(cal_event_dates)):
            cal_event_date=cal_event_dates[i]
            cal_event_id=self.frame_calendar.calevent_create(date=datetime.strptime(cal_event_date, fmt), text="" if text is None else text[i], tags=cal_event_date)
            self.cal_event_ids.append(cal_event_id)
            self.frame_calendar.tag_config(tag=cal_event_date, background='lightpink', foreground='black')
    
    def get_cal_event_ids(self, date):
        # Devuelve los ids
        fmt="%Y-%m-%d"
        return self.frame_calendar.get_calevents(date=datetime.strptime(date, fmt))

    def detele_cal_event_by_id(self, cal_event_id):
        if cal_event_id in self.cal_event_ids:
            self.frame_calendar.calevent_remove(cal_event_id)
            self.cal_event_dates.pop(self.cal_event_ids.index(cal_event_id))
            self.cal_event_ids.remove(cal_event_id)

    def delete_cal_event_by_dates(self, cal_event_dates):
        cal_event_date_index_list=[]
        cal_event_id_list=[]
        for cal_event_date in cal_event_dates:
            if cal_event_date in self.cal_event_dates:
                cal_event_date_index=self.cal_event_dates.index(cal_event_date)
                cal_event_id=self.cal_event_ids[cal_event_date_index]
                cal_event_date_index_list.append(cal_event_date_index)
                cal_event_id_list.append(cal_event_id)
                self.frame_calendar.calevent_remove(cal_event_id)
        if len(cal_event_date_index_list) > 0 and len(cal_event_id_list) > 0:
            temp_cal_event_dates=[]
            temp_cal_event_ids=[]
            for i in range(len(self.cal_event_dates)):
                if i not in cal_event_date_index_list:
                    temp_cal_event_dates.append(self.cal_event_dates[i])
                    temp_cal_event_ids.append(self.cal_event_ids[i])
            self.cal_event_dates=temp_cal_event_dates
            self.cal_event_ids=temp_cal_event_ids

    def calendar_trace_callback(self, var, index, mode):
        date=self.get_date()
        self.var_date_visualization.set(value=date)
        if self.calendar_callback is not None:
            self.calendar_callback(date=date)

    def set_date(self, date):
        self.var_date_visualization.set(value=date)
        self.var_calendar.set(value=date)

    def get_date(self):
        return self.var_calendar.get()
    
class FrameDateEntry(CreateFrame):
    def __init__(self, master, initial_date=utils_database.get_date_today().split(" ")[0], **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs) 
        self.var_date_entry=ctk.StringVar(value=initial_date)
        self.font_date_entry=ctk.CTkFont(family="Modern", size=14, weight="bold")

        self.date_entry=DateEntry(master=self, textvariable=self.var_date_entry, selectmode="day", year=int(initial_date.split("-")[0]), month=int(initial_date.split("-")[1]), day=int(initial_date.split("-")[2]), date_pattern="y-mm-dd", font=self.font_date_entry)

        self.insert_element(cad_pos="0,0", element=self.date_entry, padx=5, pady=5, sticky="")

    def set_date(self, date):
        self.date_entry.set_date(date=date)

    def get_date(self):
        return self.var_date_entry.get()
    
class FrameHourEntry(CreateFrame):
    def __init__(self, master, initial_hour=None, interval=(7,22), **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs) 
        self.var_hour_entry=ctk.StringVar(value="07:00:00" if initial_hour is None else "{}:00:00".format(str(initial_hour).zfill(2)))

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_option_menu(master=self, weight="bold", size=14, variable=self.var_hour_entry, values=["{}:00:00".format(str(val).zfill(2)) for val in range(interval[0], interval[1]+1)]), padx=5, pady=5, sticky="")

    def set_hour(self, hour):
        self.var_hour_entry.set(value="{}:00:00".format(str(hour).zfill(2)))

    def get_hour(self):
        return self.var_hour_entry.get()
    
class FrameTableDB(CreateFrame):
    def __init__(self, master, registries=None, _ids=None, frame_search_width=200, table_width=250, table_height=250, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(4,2), arr=None), **kwargs)
        self.registries=registries
        self._ids=_ids
        self.frame_search_width=frame_search_width
        self.table_width=table_width
        self.table_height=table_height
        self.frame_table=None
        self.dict_item_indexes={}
        self.var_entry=ctk.StringVar(value="")
        self.var_entry.trace_add("write", self.trace_callback)

        self.frame_search=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        self.frame_search.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_search, fg_color="transparent", img_name="search"), padx=5, pady=5, sticky="")
        self.frame_search.insert_element(cad_pos="1,0", element=FrameHelp.create_entry(master=self.frame_search, textvariable=self.var_entry, justify=ctk.LEFT, width=self.frame_search_width), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,0", element=self.frame_search, padx=5, pady=5, sticky="")

        if self.registries is not None and self._ids is not None:
            self.build_table(registries=self.registries, _ids=self._ids)
        else:
            self.build_table(registries=[], _ids=[])

    def build_table(self, registries, _ids):
        self.registries=registries
        self._ids=_ids
        self.dict_item_indexes={}
        grid_frame=GridFrame(dim=(len(self.registries), 1), arr=None)
        if self.frame_table is None:
            # Se crea la tabla 
            self.frame_table=CreateScrollableFrame(master=self, grid_frame=grid_frame, width=self.table_width, height=self.table_height)
        else:
            # Se elimina todo el contenido de la tabla y se vuelve a utilizar 
            self.frame_table.destroy_all()
            self.frame_table.create_specific_grid_frame(grid_frame=grid_frame)
        for i in range(len(self.registries)):
            registry=self.registries[i]
            self.dict_item_indexes[self._ids[i]]=i
            self.frame_table.insert_element(cad_pos="{},0".format(i), element=self.create_table_item(master=self.frame_table, registry=registry), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=self.frame_table, padx=5, pady=5, sticky="")

    def insert_registry(self, registry, _id):
        self.frame_table.insert_element_at_end(element=self.create_table_item(master=self.frame_table, registry=registry), padx=5, pady=5, sticky="ew")
        self.dict_item_indexes[_id]=self.frame_table.grid_frame.dim[0]-1
        self.registries.append(registry)
        self._ids.append(_id)

    def update_registry(self, registry, _id):
        if _id in self.dict_item_indexes.keys():
            index=self.dict_item_indexes[_id]
            table_item=self.frame_table.get_element(cad_pos="{},0".format(index)).get_element(cad_pos="0,0")
            self.update_table_item(table_item=table_item, old_registry=self.registries[self._ids.index(_id)], new_registry=registry)
            self.registries[self._ids.index(_id)]=registry

    def delete_registry(self, _id):
        if _id in self.dict_item_indexes.keys():
            index=self.dict_item_indexes[_id]
            self.frame_table.destroy_element(cad_pos="{},0".format(index))
            del self.dict_item_indexes[_id]
            self.registries.pop(self._ids.index(_id)) 
            self._ids.remove(_id)

    def rebuild_registry(self, _id):
        if _id in self.dict_item_indexes.keys():
            index=self.dict_item_indexes[_id]
            self.frame_table.destroy_element(cad_pos="{},0".format(index))
            self.frame_table.insert_element(cad_pos="{},0".format(index), element=self.create_table_item(master=self.frame_table, registry=self.registries[self._ids.index(_id)]), padx=5, pady=5, sticky="ew")

    # Este metodo se debe sobreescribir
    def update_table_item(self, table_item, old_registry, new_registry):
        pass

    # Este metodo se debe sobreescribir 
    def create_table_item(self, master, registry):
        return CreateFrame(master=master)

    # Este metodo se debe sobreescribir 
    def search_compare(self, registry):
        return ""
    
    def trace_callback(self, var, index, mode): 
        name=self.var_entry.get()
        for i in range(len(self.registries)):
            registry=self.registries[i]
            index=self.dict_item_indexes[self._ids[i]]
            elem=self.frame_table.get_element(cad_pos="{},0".format(index))
            if re.search(name.lower(), self.search_compare(registry=registry).lower()) is None:
                elem.hide_frame()
            else:
                elem.show_frame()        

class FrameButtonsBasicOperation(CreateFrame):
    def __init__(self, master, button_create_new_registry, button_save_registry, button_close, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(4,1), arr=None), **kwargs)
        self.var_label=ctk.StringVar(value="")

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=self, command=button_create_new_registry, fg_color="lightcoral", text="Crear nuevo registro", compound=ctk.LEFT, img_name="create", img_width=20, img_height=20), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=self, command=button_save_registry, fg_color="lightcoral", text="Guardar registro", compound=ctk.LEFT, img_name="save", img_width=20, img_height=20), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=self, command=button_close, fg_color="lightcoral", text="Cerrar", compound=ctk.LEFT, img_name="remove", img_width=20, img_height=20), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_label(master=self, weight="bold", fg_color="coral", textvariable=self.var_label), padx=5, pady=5, sticky="")

    def clear_label(self):
        self.var_label.set(value="")

    def set_label(self, value):
        self.var_label.set(value=value)

class FrameEntryAutocomplete(CreateFrame):
    def __init__(self, master, var_entry, autocomplete_list=[], entry_width=250, entry_height=30, list_box_height=100, **kwargs):
        CreateFrame.__init__(self, master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.var_entry=var_entry
        self.var_entry.trace_add("write", self.trace_callback)
        self.autocomplete_list=autocomplete_list
        
        self.frame_list_box=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,1), arr=None), height=list_box_height)
        self.frame_list_box.enable_fixed_size()
        # self.list_box=CTkListbox(master=self.frame_list_box, width=entry_width, height=list_box_height, text_color="gray50", command=self.selected_elem, border_width=0, hover_color="gray80", select_color="gray80", hightlight_color="gray80")
        self.list_box=CTkListbox(master=self.frame_list_box, width=entry_width, height=list_box_height, text_color="gray50", command=self.selected_elem, border_width=0, hover_color="gray80")
        self.frame_list_box.insert_element(cad_pos="0,0", element=self.list_box, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_entry(master=self, textvariable=self.var_entry, justify=ctk.LEFT, width=entry_width, height=entry_height), padx=5, pady=0, sticky="")
        self.insert_element(cad_pos="1,0", element=self.frame_list_box, padx=5, pady=5, sticky="nsew")

        self.trace_callback(None, None, None)

    def selected_elem(self, elem):
        self.var_entry.set(value=elem)

    def set_autocomplete_list(self, autocomplete_list):
        self.autocomplete_list=autocomplete_list
        self.trace_callback(None, None, None)

    def trace_callback(self, var, index, mode): 
        self.list_box.delete("ALL")
        cad=self.var_entry.get()
        if cad != "":
            for i in range(len(self.autocomplete_list)):
                elem=self.autocomplete_list[i]
                if re.search(cad.lower(), elem.lower()) is not None:
                    self.list_box.insert(i, elem)    
        else:
            for i in range(len(self.autocomplete_list)):
                elem=self.autocomplete_list[i]
                self.list_box.insert(i, elem)

class FrameTakePhoto(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(4,2), arr=np.array([["0,0","0,1"],["1,0","1,0"],["2,0","2,0"],["3,0","3,0"]])), **kwargs)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        self.thread_camera_1=self.root.thread_camera_1
        self.thread_camera_2=self.root.thread_camera_2
        self.var_camera=ctk.StringVar(value="Selecciona una camara")
        self.var_camera.trace_add("write", self.trace_callback)
        self.image_taken=None
        self.patient_id=None
        self.after_id=None
        self.after_ms=100
        self.save_path="{}/images/patients".format(os.path.dirname(os.path.abspath('__file__')))

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text=""), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,1", element=FrameHelp.create_label(master=self, text=""), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_option_menu(master=self, variable=self.var_camera, values=["Selecciona una camara", "Camara 1", "Camara 2"]), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=self, text="Tomar foto", fg_color="coral", img_name="take_photo", img_width=30, img_height=30, command=self.button_take_photo), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self, text="Eliminar foto", fg_color="coral", img_name="remove", img_width=30, img_height=30, command=self.button_delete_photo), padx=5, pady=5, sticky="")

        self.hide_element(cad_pos="0,1")

    def clear_image(self):
        image=ui_images.get_image(name="no_user", width=250, height=250)
        self.get_element(cad_pos="0,0").configure(image=image)
        self.get_element(cad_pos="0,0").image=image

    def set_image(self, patient_id):
        self.patient_id=patient_id
        image=ui_images.get_patient_photo(patient_id=patient_id, width=250, height=250)
        self.get_element(cad_pos="0,0").configure(image=image)
        self.get_element(cad_pos="0,0").image=image

    def trace_callback(self, var, index, mode):
        self.cancel_show_camera_view()
        if self.var_camera.get() == "Selecciona una camara":
            self.hide_element(cad_pos="0,1")
        elif self.var_camera.get() == "Camara 1":
            self.show_camera_view(thread_camera=self.thread_camera_1)
            self.show_element(cad_pos="0,1")
        elif self.var_camera.get() == "Camara 2":
            self.show_camera_view(thread_camera=self.thread_camera_2)
            self.show_element(cad_pos="0,1")
        
    def cancel_show_camera_view(self):
        if self.after_id is not None:
            self.after_cancel(id=self.after_id)
            self.after_id=None

    def button_take_photo(self):
        if self.var_camera.get() != "Selecciona una camara":
            image,frame=self.get_image_by_frame(frame=self.thread_camera_1.frame if self.var_camera.get() == "Camara 1" else self.thread_camera_2.frame)
            if frame is not None:
                self.image_taken=frame
                self.get_element(cad_pos="0,0").configure(image=image)
                self.get_element(cad_pos="0,0").image=image
                self.var_camera.set(value="Selecciona una camara")
    
    def button_delete_photo(self):
        if tk.messagebox.askyesnocancel(title="Eliminar foto", message="¿Esta seguro de eliminar la foto del paciente?", parent=self):
            self.image_taken=None
            self.clear_image()

    def get_image_by_frame(self, frame):
        if frame is None:
            image=ui_images.get_image(name="no_image_available", width=250, height=250)
            return image, None
        else:
            temp_frame=utils.resize_image(scale_percent=50, img=frame.copy())
            image=utils.frame_to_img(temp_frame)
            return image, temp_frame

    def show_camera_view(self, thread_camera):
        image,_=self.get_image_by_frame(frame=thread_camera.frame)
        self.get_element(cad_pos="0,1").configure(image=image)
        self.get_element(cad_pos="0,1").image=image
        self.after_id=self.after(self.after_ms, lambda: self.show_camera_view(thread_camera=thread_camera))

    def save_photo(self, patient_id):
        if self.image_taken is not None:
            path="{}/{}".format(self.save_path, str(patient_id))
            if not os.path.exists(path):
                os.mkdir(path)
            path="{}/{}.png".format(path, str(patient_id))
            cv2.imwrite(path, cv2.cvtColor(self.image_taken, cv2.COLOR_RGB2BGR))
            return True
        else:
            return False
        
    @staticmethod
    def delete_photo(patient_id):
        save_path="{}/images/patients".format(os.path.dirname(os.path.abspath('__file__')))
        path="{}/{}".format(save_path, str(patient_id))
        if os.path.exists(path):
            shutil.rmtree(path)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FrameLeft(CreateScrollableFrame, BuildComponent):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(8,1), arr=None), **kwargs)
        BuildComponent.__init__(self)
        
        self.build_component()

    def build_component(self):
        self.destroy_all()
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Notificaciones base de datos", weight="bold", size=25, fg_color="coral"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameNotifyDBChanges(master=self, fg_color="pink", name="FrameNotifyDBChanges"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self, text="Limpieza de agenda", weight="bold", size=25, fg_color="coral"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameScheduleCleaning(master=self, fg_color="pink", name="FrameScheduleCleaning"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_label(master=self, text="Pacientes", weight="bold", size=25, fg_color="coral"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="5,0", element=FramePatientBasicOperations(master=self, fg_color="pink", frame_search_width=250, table_width=800, table_height=600, name="FramePatientBasicOperations"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="6,0", element=FrameHelp.create_label(master=self, text="Calendario", weight="bold", size=25, fg_color="coral"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="7,0", element=FrameCalendar(master=self, fg_color="pink", name="FrameCalendar"), padx=5, pady=100, sticky="")

class FrameCollectionNotification(CreateFrame, BuildComponent):
    def __init__(self,collection_name, text, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(4,1), arr=None), **kwargs)
        BuildComponent.__init__(self)
        self.collection_name=collection_name
        self.var_insert=ctk.StringVar(value="")
        self.var_update=ctk.StringVar(value="")
        self.var_delete=ctk.StringVar(value="")
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text=text, size=14, weight="bold"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self, textvariable=self.var_insert, text="", size=12, anchor="w"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self, textvariable=self.var_update, text="", size=12, anchor="w"), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_label(master=self, textvariable=self.var_delete, text="", size=12, anchor="w"), padx=5, pady=5, sticky="ew")

        self.build_component()

    def build_component(self):
        db_notifications=self.primary_connection_db.db.db_notifications[self.collection_name]
        for op,var,cad_pos,t_text in [("insert",self.var_insert,"1,0",("inserciones","insercion(es)")),("update",self.var_update,"2,0",("modificaciones","modificacion(es)")),("delete",self.var_delete,"3,0",("eliminaciones","eliminacion(es)"))]:
            if len(db_notifications[op]) == 0:
                var.set(value="Por el momento no hay {}".format(t_text[0]))
                self.get_element(cad_pos=cad_pos).configure(text_color="black")
            else:
                var.set(value="Hay {} {}".format(len(db_notifications[op]),t_text[1]))
                self.get_element(cad_pos=cad_pos).configure(text_color="red")

class FrameNotifyDBChanges(CreateFrame, BuildComponent, BuildDBComponent):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,5), arr=np.array([["0,0","0,0","0,0","0,0","0,4"],["0,0","0,0","0,0","0,0","1,4"],["2,0","2,0","2,0","2,0","2,0"]])), **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.frame_notification=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,2), arr=np.array([["0,0","0,0"],["1,0","1,1"],["2,0","2,0"]])))
        self.frame_database_updates=CreateFrame(master=self.frame_notification, grid_frame=GridFrame(dim=(2,1), arr=None))
        self.frame_database_updates.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_database_updates, text="Ultima fecha de actualizacion de la base de datos", size=16, weight="bold"), padx=5, pady=5, sticky="")
        self.frame_database_updates.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self.frame_database_updates, text="{}".format(self.primary_connection_db.db.collections["DatabaseUpdates"].database_update_date), size=12), padx=5, pady=5, sticky="")
        self.frame_notification.insert_element(cad_pos="0,0", element=self.frame_database_updates, padx=5, pady=5, sticky="")
        self.frame_notification.insert_element(cad_pos="1,0", element=FrameCollectionNotification(master=self.frame_notification, collection_name="Pathology", text="Patologias", name="FrameCollectionNotificationPathology"), padx=5, pady=5, sticky="")
        self.frame_notification.insert_element(cad_pos="1,1", element=FrameCollectionNotification(master=self.frame_notification, collection_name="Disease", text="Enfermedades", name="FrameCollectionNotificationDisease"), padx=5, pady=5, sticky="")
        self.frame_notification.insert_element(cad_pos="2,0", element=FrameCollectionNotification(master=self.frame_notification, collection_name="Patient", text="Pacientes", name="FrameCollectionNotificationPatient"), padx=5, pady=5, sticky="")
        
        self.frame_db_information=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,1), arr=None))
        self.frame_db_information.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_db_information, text="Base de datos".upper(), size=18, weight="bold", fg_color="transparent"), padx=5, pady=5, sticky="ew")
        self.frame_db_information.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self.frame_db_information, text="'{}'".format(self.primary_connection_db.connection_type.upper()), size=14, weight="bold", fg_color="transparent"), padx=5, pady=5, sticky="ew")
        self.frame_db_information.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self.frame_db_information, text="", img_name="local_database" if self.primary_connection_db.connection_type == 'local' else "cloud_database", img_width=80, img_height=80, fg_color="transparent"), padx=5, pady=5, sticky="ew")
        
        self.frame_load_database=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,4), arr=np.array([["0,0","0,0","0,0","0,0"],["1,0","1,1","1,2","1,3"],["1,0","1,1","1,2","1,3"]])))
        self.frame_load_database.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_load_database, text="Cargar base de datos", weight="bold", size=16), padx=5, pady=5, sticky="ew")
        self.frame_load_database.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=self.frame_load_database, text="", img_name="local_database", img_width=80, img_height=80, fg_color="transparent"), padx=5, pady=5, sticky="")
        self.frame_load_database.insert_element(cad_pos="1,1", element=FrameHelp.create_label(master=self.frame_load_database, text="", img_name="arrows", img_width=50, img_height=50, fg_color="transparent"), padx=5, pady=5, sticky="")
        self.frame_load_database.insert_element(cad_pos="1,2", element=FrameHelp.create_label(master=self.frame_load_database, text="", img_name="cloud_database", img_width=80, img_height=80, fg_color="transparent"), padx=5, pady=5, sticky="")
        self.frame_load_database.insert_element(cad_pos="1,3", element=FrameHelp.create_button(master=self.frame_load_database, height=60, command=self.button_load_database, text="Cargar la base de datos '{}' en la base de datos '{}'".format(self.primary_connection_db.connection_type, 'local' if self.primary_connection_db.connection_type == 'cloud' else 'cloud'), weight="bold", size=12, fg_color="lightcoral"), padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=self.frame_notification, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,4", element=self.frame_db_information, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,4", element=FrameHelp.create_button(master=self, text="Guardar en base de datos", command=self.button_save, weight="bold", img_name="save", img_width=50, img_height=50, fg_color="lightcoral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=self.frame_load_database, padx=5, pady=5, sticky="ew")

        self.build_component()

    def build_component(self):
        pass

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Pathology":
            frame_collection_notification: FrameCollectionNotification=self.frame_notification.get_element(cad_pos="1,0")
            frame_collection_notification.build_component()
        elif collection_name == "Disease":
            frame_collection_notification: FrameCollectionNotification=self.frame_notification.get_element(cad_pos="1,1")
            frame_collection_notification.build_component()
        elif collection_name == "Patient":
            frame_collection_notification: FrameCollectionNotification=self.frame_notification.get_element(cad_pos="2,0")
            frame_collection_notification.build_component()

    def button_save(self):
        if self.primary_connection_db.db.is_empty_db_notifications():
            tk.messagebox.showinfo(title="Guardar en base de datos", message="No hay cambios por guardar en la base de datos. Todo esta actualizado.", parent=self)
        else:
            if tk.messagebox.askyesnocancel(title="Guardar en base de datos", message="¿Esta seguro de guardar los cambios pendientes en la base de datos '{}'?".format(self.primary_connection_db.connection_type), parent=self):
                self.primary_connection_db.operation(op="save")
                if self.primary_connection_db.message_notification_connection_db.message_type == MessageTypesConnectionDB.SAVE_OK:
                    for cad_pos in ["1,0","1,1","2,0"]:
                        frame_collection_notification: FrameCollectionNotification=self.frame_notification.get_element(cad_pos=cad_pos)
                        frame_collection_notification.build_component()
                tk.messagebox.showinfo(title=self.primary_connection_db.message_notification_connection_db.message, message=self.primary_connection_db.message_notification_connection_db.specific_message, parent=self)
          
    def button_load_database(self):
        if tk.messagebox.askyesnocancel(title="Cargar base de datos", message="¿Esta seguro de cargar la base de datos '{}' en la base de datos '{}'?".format(self.primary_connection_db.connection_type, 'local' if self.primary_connection_db.connection_type == 'cloud' else 'cloud'), parent=self):
            username,password=self.request_username_password()
            if username is not None and password is not None:
                self.primary_connection_db.operation(op="load_database", username=username, password=password)
                tk.messagebox.showinfo(title=self.primary_connection_db.message_notification_connection_db.message, message=self.primary_connection_db.message_notification_connection_db.specific_message, parent=self)

    def request_username_password(self):
        username=tk.simpledialog.askstring("Ingrese el nombre de usuario", "Usuario:")
        if username is not None:
            password=tk.simpledialog.askstring("Ingrese la contraseña de usuario", "Contraseña:", show='*')
            if password is not None:
                return username, password
            else:
                return username, None
        else:
            return None, None

class FrameScheduleCleaning(CreateFrame, BuildComponent, BuildDBComponent):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,3), arr=np.array([["0,0","0,1","0,2"],["1,0","1,0","1,0"]])), **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        self.dict_patients_involved={}
        self.var_option=ctk.StringVar(value="1 semana")
        self.var_option.trace_add("write", self.trace_callback)

        values=["1 semana", "1 mes", "2 meses", "3 meses", "4 meses", "5 meses", "6 meses"]
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_table_sub_item(master=self, title="Total de citas a eliminar", text="", size=14, width=250, height=100), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,1", element=FrameHelp.create_option_menu(master=self, size=12, weight="bold", variable=self.var_option, values=values), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="0,2", element=FrameHelp.create_button(master=self, text="Eliminar", size=14, weight="bold", img_name="remove", img_width=50, img_height=50, fg_color="transparent", command=self.button_delete), padx=5, pady=5, sticky="")
        
        self.build_component()
        self.update_info()

    def build_component(self):
        pass

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Patient":
            self.update_info()

    def trace_callback(self, var, index, mode):
        self.update_info()

    def button_delete(self):
        if len(self.dict_patients_involved) == 0:
            tk.messagebox.showinfo(title="Eliminar citas", message="No hay citas para eliminar.", parent=self)
        else:
            if tk.messagebox.askyesnocancel(title="Eliminar citas", message="¿Esta seguro de quedarse solo con las citas de hace '{}'?".format(self.var_option.get()), parent=self):
                for patient_id in self.dict_patients_involved.keys():
                    updated_patient=self.primary_connection_db.collection_notification_update(collection_name="Patient", obj_update={"_id": patient_id, "update": {"schedule": self.dict_patients_involved[patient_id]["filtered_schedule"]}})
                    self.root.notify_build_db_component(collection_name="Patient", registry=updated_patient, op="update", notify_build_db_component_names=["FramePatientBasicOperations", "FrameNotifyDBChanges", "FrameAuthenticatedPatientTable", "FrameCalendar", "FrameCalendarSpecificDate"], old_registry=self.dict_patients_involved[patient_id]["patient"])

                self.root.get_child(name="FrameCalendar").restart_calendar()
                self.update_info()

    def update_info(self):
        self.dict_patients_involved={}
        quantity=7 if self.var_option.get() == "1 semana" else int(30 * int(self.var_option.get().split(" ")[0]))
        total=0
        patient_list=self.primary_connection_db.db.collections["Patient"]
        for patient in patient_list:
            patient: Patient
            filtered_schedule=[]
            count=0
            for date_hour in patient.schedule:
                date,hour=date_hour.split(" ")
                if utils_database.diff_days(date=date) < quantity:
                    filtered_schedule.append(date_hour)
                else:
                    count+=1
            total+=count
            if count > 0:
                self.dict_patients_involved[patient._id]={
                    "patient": patient,
                    "filtered_schedule": filtered_schedule,
                    "count": count
                }

        self.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(text=str(total))
        self.destroy_element(cad_pos="1,0")
        l=[]
        for patient_id in self.dict_patients_involved.keys():
            l.append("Paciente: {}\nCitas a eliminar: {}".format(self.dict_patients_involved[patient_id]["patient"].name, self.dict_patients_involved[patient_id]["count"]))
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_table_list_sub_item(master=self, title="Informacion especifica", l=l, width=600, height=250), padx=5, pady=5, sticky="")

class FramePathologyBasicOperations(FrameTableDB, BuildComponent, BuildDBComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, sub_frame_width, sub_frame_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.insert_element(cad_pos="0,1", element=FrameButtonsBasicOperation(master=self, button_create_new_registry=self.button_create_new_registry, button_save_registry=self.button_save_registry, button_close=self.button_close), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=FramePathology(master=self, width=sub_frame_width, height=sub_frame_height), padx=5, pady=5, sticky="").hide_frame()

        self.build_component()

    def build_component(self):
        self.build_table(registries=[], _ids=[])

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Pathology":
            # Actualizamos la tabla
            if op == "insert": self.insert_registry(registry=registry, _id=str(registry._id))
            elif op == "update": self.update_registry(registry=registry, _id=str(registry._id))
            elif op == "delete": self.delete_registry(_id=str(registry._id))

            if self.get_element(cad_pos="1,1").is_visible and registry._id == self.get_element(cad_pos="1,1").registry._id:
                self.button_close()
        
    def update_table_item(self, table_item, old_registry, new_registry):
        if new_registry.name != old_registry.name:
            table_item.get_element(cad_pos="0,1").get_element(cad_pos="1,0").configure(text=new_registry.name)
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(command=lambda: self.button_delete(registry=new_registry))
        
    def create_table_item(self, master, registry):
        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(1,5), arr=None), height=100, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(2,1), arr=None), width=70, height=150)
        buttons_table_item.enable_fixed_size()
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_delete(registry=registry), fg_color="transparent", img_name="remove", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_table_sub_item(master=table_item, title="Nombre", text=registry.name, fg_color="transparent", width=200, height=80), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def search_compare(self, registry):
        return registry.name
    
    def button_select(self, registry):
        self.get_element(cad_pos="0,1").set_label(value="Registro seleccionado")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=registry)

    def button_delete(self, registry):
        if tk.messagebox.askyesnocancel(title="Eliminar registro", message="¿Esta seguro de eliminar el registro '{}'?".format(registry.name), parent=self):
            self.root.notify_build_db_component(collection_name="Pathology", registry=registry, op="delete", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])

    def button_create_new_registry(self):
        self.get_element(cad_pos="0,1").set_label(value="Registro nuevo en proceso...")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=Pathology.get_empty())

    def button_save_registry(self):
        if self.get_element(cad_pos="1,1").is_visible:
            new_registry=self.get_element(cad_pos="1,1").get_registry()
            old_registry=self.get_element(cad_pos="1,1").registry
            if self.registry_verification(registry=new_registry):
                if old_registry.name != "":
                    self.root.notify_build_db_component(collection_name="Pathology", registry=new_registry, op="update", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])
                else:
                    self.root.notify_build_db_component(collection_name="Pathology", registry=new_registry, op="insert", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])
        else:
            tk.messagebox.showinfo(title="Guardar registro", message="No hay nada para guardar.", parent=self)

    def button_close(self):
        self.get_element(cad_pos="0,1").clear_label()
        self.hide_element(cad_pos="1,1")

    def registry_verification(self, registry):
        if registry is not None:
            if registry.name != "":
                if registry.name.lower() not in list(map(lambda elem: elem.name.lower(), self.registries)):
                    return True
                else:
                    tk.messagebox.showinfo(title="Registro repetido", message="El nombre ingresado ya existe.", parent=self)
                    return False
            else:
                tk.messagebox.showinfo(title="Registro no valido", message="El nombre no debe estar vacio.", parent=self)
                return False
        return False

class FramePathology(CreateScrollableFrame, TemplateFrameClasses):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        TemplateFrameClasses.__init__(self)
        self.registry=None
        self.var_pathology_name=ctk.StringVar(value="")
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        
        autocomplete_list=list(map(lambda elem: elem.name, self.primary_connection_db.db.collections["Pathology"]))
        self.insert_element(cad_pos="0,0", element=FrameEntryAutocomplete(master=self, var_entry=self.var_pathology_name, autocomplete_list=autocomplete_list, entry_width=200, entry_height=30, list_box_height=100), padx=5, pady=5, sticky="")

    def clear_all(self):
        self.registry=None
        self.var_pathology_name.set(value="")

    def set_registry(self, registry):
        self.clear_all()
        self.registry=registry

        name=self.registry.name
        
        self.var_pathology_name.set(value=name)

    def get_registry(self):
        if self.registry is not None:
            return Pathology(_id=self.registry._id, name=self.var_pathology_name.get())
        return None

class FrameDiseaseBasicOperations(FrameTableDB, BuildComponent, BuildDBComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, sub_frame_width, sub_frame_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.insert_element(cad_pos="0,1", element=FrameButtonsBasicOperation(master=self, button_create_new_registry=self.button_create_new_registry, button_save_registry=self.button_save_registry, button_close=self.button_close), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=FrameDisease(master=self, width=sub_frame_width, height=sub_frame_height), padx=5, pady=5, sticky="").hide_frame()

        self.build_component()

    def build_component(self):
        self.build_table(registries=[], _ids=[])

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Disease":
            # Actualizamos la tabla
            if op == "insert": self.insert_registry(registry=registry, _id=str(registry._id))
            elif op == "update": self.update_registry(registry=registry, _id=str(registry._id))
            elif op == "delete": self.delete_registry(_id=str(registry._id))

            if self.get_element(cad_pos="1,1").is_visible and registry._id == self.get_element(cad_pos="1,1").registry._id:
                self.button_close()
        
    def update_table_item(self, table_item, old_registry, new_registry):
        if new_registry.name != old_registry.name:
            table_item.get_element(cad_pos="0,1").get_element(cad_pos="1,0").configure(text=new_registry.name)
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(command=lambda: self.button_delete(registry=new_registry))
        
    def create_table_item(self, master, registry):
        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(1,5), arr=None), height=100, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(2,1), arr=None), width=70, height=150)
        buttons_table_item.enable_fixed_size()
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_delete(registry=registry), fg_color="transparent", img_name="remove", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_table_sub_item(master=table_item, title="Nombre", text=registry.name, fg_color="transparent", width=200, height=80), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def search_compare(self, registry):
        return registry.name
    
    def button_select(self, registry):
        self.get_element(cad_pos="0,1").set_label(value="Registro seleccionado")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=registry)

    def button_delete(self, registry):
        if tk.messagebox.askyesnocancel(title="Eliminar registro", message="¿Esta seguro de eliminar el registro '{}'?".format(registry.name), parent=self):
            self.root.notify_build_db_component(collection_name="Disease", registry=registry, op="delete", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])

    def button_create_new_registry(self):
        self.get_element(cad_pos="0,1").set_label(value="Registro nuevo en proceso...")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=Disease.get_empty())

    def button_save_registry(self):
        if self.get_element(cad_pos="1,1").is_visible:
            new_registry=self.get_element(cad_pos="1,1").get_registry()
            old_registry=self.get_element(cad_pos="1,1").registry
            if self.registry_verification(registry=new_registry):
                if old_registry.name != "":
                    self.root.notify_build_db_component(collection_name="Disease", registry=new_registry, op="update", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])
                else:
                    self.root.notify_build_db_component(collection_name="Disease", registry=new_registry, op="insert", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges"])
        else:
            tk.messagebox.showinfo(title="Guardar registro", message="No hay nada para guardar.", parent=self)

    def button_close(self):
        self.get_element(cad_pos="0,1").clear_label()
        self.hide_element(cad_pos="1,1")

    def registry_verification(self, registry):
        if registry is not None:
            if registry.name != "":
                if registry.name.lower() not in list(map(lambda elem: elem.name.lower(), self.registries)):
                    return True
                else:
                    tk.messagebox.showinfo(title="Registro repetido", message="El nombre ingresado ya existe.", parent=self)
                    return False
            else:
                tk.messagebox.showinfo(title="Registro no valido", message="El nombre no debe estar vacio.", parent=self)
                return False
        return False

class FrameDisease(CreateScrollableFrame, TemplateFrameClasses):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        TemplateFrameClasses.__init__(self)
        self.registry=None
        self.var_disease_name=ctk.StringVar(value="")
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        
        autocomplete_list=list(map(lambda elem: elem.name, self.primary_connection_db.db.collections["Disease"]))
        self.insert_element(cad_pos="0,0", element=FrameEntryAutocomplete(master=self, var_entry=self.var_disease_name, autocomplete_list=autocomplete_list, entry_width=200, entry_height=30, list_box_height=100), padx=5, pady=5, sticky="")

    def clear_all(self):
        self.registry=None
        self.var_disease_name.set(value="")

    def set_registry(self, registry):
        self.clear_all()
        self.registry=registry

        name=self.registry.name
        
        self.var_disease_name.set(value=name)

    def get_registry(self):
        if self.registry is not None:
            return Disease(_id=self.registry._id, name=self.var_disease_name.get())
        return None

class FrameScheduleBasicOperations(FrameTableDB, BuildComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, sub_frame_width, sub_frame_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)

        self.insert_element(cad_pos="0,1", element=FrameButtonsBasicOperation(master=self, button_create_new_registry=self.button_create_new_registry, button_save_registry=self.button_save_registry, button_close=self.button_close), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=FrameSchedule(master=self, width=sub_frame_width, height=sub_frame_height), padx=5, pady=5, sticky="").hide_frame()

        self.build_component()

    def build_component(self):
        pass

    def update_table_item(self, table_item, old_registry, new_registry):
        if new_registry != old_registry:
            table_item.get_element(cad_pos="0,1").get_element(cad_pos="1,0").configure(text=new_registry)
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(command=lambda: self.button_delete(registry=new_registry))

    def create_table_item(self, master, registry):
        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(1,5), arr=None), height=100, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(2,1), arr=None), width=70, height=150)
        buttons_table_item.enable_fixed_size()
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_delete(registry=registry), fg_color="transparent", img_name="remove", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_table_sub_item(master=table_item, title="Fecha / Hora", text=registry, size=10, fg_color="transparent", width=150, height=80), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def search_compare(self, registry):
        return registry
    
    def button_select(self, registry):
        self.get_element(cad_pos="0,1").set_label(value="Registro seleccionado")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=registry)

    def button_delete(self, registry):
        if tk.messagebox.askyesnocancel(title="Eliminar registro", message="¿Esta seguro de eliminar el registro '{}'?".format(registry), parent=self):
            self.delete_registry(_id=self._ids[self.registries.index(registry)])
            if self.get_element(cad_pos="1,1").is_visible and self.get_element(cad_pos="1,1").registry == registry:
                self.button_close()

    def button_create_new_registry(self):
        self.get_element(cad_pos="0,1").set_label(value="Registro nuevo en proceso...")
        self.show_element(cad_pos="1,1")
        self.get_element(cad_pos="1,1").set_registry(registry=None)

    def button_save_registry(self):
        if self.get_element(cad_pos="1,1").is_visible:
            new_registry=self.get_element(cad_pos="1,1").get_registry()
            old_registry=self.get_element(cad_pos="1,1").registry
            if self.registry_verification(registry=new_registry):
                if old_registry is None:
                    # Registro nuevo
                    self.insert_registry(registry=new_registry, _id=ObjectId())
                else:
                    # Registro existente actualizado
                    self.update_registry(registry=new_registry, _id=self._ids[self.registries.index(old_registry)])
                self.button_close()
        else:
            tk.messagebox.showinfo(title="Guardar registro", message="No hay nada para guardar.", parent=self)

    def button_close(self):
        self.get_element(cad_pos="0,1").clear_label()
        self.hide_element(cad_pos="1,1")

    def registry_verification(self, registry):
        if registry is not None:
            if registry not in list(map(lambda elem: elem, self.registries)):
                return True
            else:
                date,hour=registry.split(" ")
                tk.messagebox.showinfo(title="Registro repetido", message="Ya hay una cita agendada para el dia '{}' a las '{}'.".format(date, hour), parent=self)
                return False
        return False

class FrameSchedule(CreateScrollableFrame, TemplateFrameClasses):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        TemplateFrameClasses.__init__(self)
        self.registry=None

        self.insert_element(cad_pos="0,0", element=FrameCalendarDate(master=self, calendar_text_size=10, calendar_visualization_text_size=12, calendar_width=250, calendar_height=250, calendar_callback=None), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameHourEntry(master=self, initial_hour=None, interval=(7,22)), padx=5, pady=5, sticky="")
        
    def clear_all(self):
        self.registry=None

    def set_registry(self, registry):
        self.clear_all()
        self.registry=registry

        self.get_element(cad_pos="0,0").set_date(date=utils_database.get_date_today().split(" ")[0] if self.registry is None else registry.split(" ")[0])
        self.get_element(cad_pos="1,0").set_hour(hour=7 if self.registry is None else int(registry.split(" ")[1].split(":")[0]))

    def get_registry(self):
        return "{} {}".format(self.get_element(cad_pos="0,0").get_date(), self.get_element(cad_pos="1,0").get_hour())

class FramePatientBasicOperations(FrameTableDB, BuildComponent, BuildDBComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        self.session_list=self.primary_connection_db.db.collections["Session"]
        self.session_names=list(map(lambda elem: elem.name, self.session_list))
        self.session_ids=list(map(lambda elem: elem._id, self.session_list))
        self.current_situation_list=self.primary_connection_db.db.collections["CurrentSituation"]
        self.current_situation_names=list(map(lambda elem: elem.name, self.current_situation_list))
        self.current_situation_ids=list(map(lambda elem: elem._id, self.current_situation_list))

        self.insert_element(cad_pos="2,0", element=FrameButtonsBasicOperation(master=self, button_create_new_registry=self.button_create_new_registry, button_save_registry=self.button_save_registry, button_close=self.button_close), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=FramePatient(master=self, width=800, height=600), padx=5, pady=5, sticky="").hide_frame()

        self.build_component()

    def build_component(self):
        registries=copy.deepcopy(self.primary_connection_db.db.collections["Patient"])
        _ids=list(map(lambda elem: str(elem._id), registries))
        self.build_table(registries=registries, _ids=_ids)

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Patient":
            # Actualizamos la tabla
            if op == "insert": self.insert_registry(registry=registry, _id=str(registry._id))
            elif op == "update": self.update_registry(registry=registry, _id=str(registry._id))
            elif op == "delete": self.delete_registry(_id=str(registry._id))

            if self.get_element(cad_pos="3,0").is_visible and registry._id == self.get_element(cad_pos="3,0").registry._id:
                self.button_close()

    def listener_db_component(self, collection_name, registry, **kwargs):
        if collection_name == "Patient":
            self.update_registry(registry=registry, _id=str(registry._id))

    def update_table_item(self, table_item, old_registry, new_registry):
        image=ui_images.get_patient_photo(patient_id=new_registry._id, width=150, height=150)
        table_item.get_element(cad_pos="0,1").configure(image=image)
        table_item.get_element(cad_pos="0,1").image=image
        if new_registry.name != old_registry.name:
            table_item.get_element(cad_pos="0,2").get_element(cad_pos="1,0").configure(text=new_registry.name)
        if new_registry.pathologies != old_registry.pathologies:
            table_item.destroy_element(cad_pos="0,3")
            pathology_names=[self.primary_connection_db.db.find_by_id(collection_name="Pathology", _id=new_registry.pathologies[i]).name for i in range(len(new_registry.pathologies))]
            table_item.insert_element(cad_pos="0,3", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Patologias", l=pathology_names, width=370, height=250), padx=5, pady=5, sticky="")
        if new_registry.diseases != old_registry.diseases:
            table_item.destroy_element(cad_pos="0,4")
            disease_names=[self.primary_connection_db.db.find_by_id(collection_name="Disease", _id=new_registry.diseases[i]).name for i in range(len(new_registry.diseases))]
            table_item.insert_element(cad_pos="0,4", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Enfermedades", l=disease_names, width=370, height=250), padx=5, pady=5, sticky="")
        if new_registry.session != old_registry.session:
            table_item.get_element(cad_pos="1,2").get_element(cad_pos="1,0").configure(text=self.session_names[self.session_ids.index(new_registry.session)])
        if new_registry.current_situation != old_registry.current_situation:
            current_situation_color=self.current_situation_list[self.current_situation_ids.index(new_registry.current_situation)].color
            table_item.get_element(cad_pos="1,3").configure(fg_color=utils.rgb_to_hex(color=current_situation_color))
            table_item.get_element(cad_pos="1,3").get_element(cad_pos="1,0").configure(text=self.current_situation_names[self.current_situation_ids.index(new_registry.current_situation)])
        if new_registry.schedule != old_registry.schedule:
            table_item.destroy_element(cad_pos="1,4")
            table_item.insert_element(cad_pos="1,4", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Agenda", l=new_registry.schedule, width=370, height=250), padx=5, pady=5, sticky="")
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(command=lambda: self.button_delete(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="2,0").configure(command=lambda: self.button_authentication(registry=new_registry))
        
    def create_table_item(self, master, registry):
        pathology_names=[self.primary_connection_db.db.find_by_id(collection_name="Pathology", _id=registry.pathologies[i]).name for i in range(len(registry.pathologies))]
        disease_names=[self.primary_connection_db.db.find_by_id(collection_name="Disease", _id=registry.diseases[i]).name for i in range(len(registry.diseases))]
        var_session_name=ctk.StringVar(value=self.session_names[self.session_ids.index(registry.session)])
        var_current_situation_name=ctk.StringVar(value=self.current_situation_names[self.current_situation_ids.index(registry.current_situation)])
        current_situation_color=self.current_situation_list[self.current_situation_ids.index(registry.current_situation)].color
        schedule=copy.deepcopy(registry.schedule)
        schedule.sort(reverse=True)

        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(2,5), arr=np.array([["0,0","0,1","0,2","0,3","0,4"],["0,0","0,1","1,2","1,3","1,4"]])), height=250, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(3,1), arr=None), width=70, height=250)
        buttons_table_item.enable_fixed_size()
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_delete(registry=registry), fg_color="transparent", img_name="remove", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_authentication(registry=registry), fg_color="transparent", img_name="authentication", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_photo_item(master=table_item, patient_id=registry._id, photo_width=150, photo_height=150), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,2", element=FrameHelp.create_table_sub_item(master=table_item, title="Nombre", text=registry.name, fg_color="transparent", width=200, height=80), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,3", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Patologias", l=pathology_names, width=370, height=250), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,4", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Enfermedades", l=disease_names, width=370, height=250), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="1,2", element=FrameHelp.create_table_sub_item(master=table_item, title="Session", text=var_session_name.get(), fg_color="transparent", width=150, height=80), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="1,3", element=FrameHelp.create_table_sub_item(master=table_item, title="Situacion actual", text=var_current_situation_name.get(), fg_color=utils.rgb_to_hex(color=current_situation_color), width=250, height=80), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="1,4", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Agenda", l=schedule, width=370, height=250), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def search_compare(self, registry):
        return registry.name
    
    def button_select(self, registry):
        self.get_element(cad_pos="2,0").set_label(value="Registro seleccionado")
        self.show_element(cad_pos="3,0")
        self.get_element(cad_pos="3,0").set_registry(registry=registry)

    def button_delete(self, registry):
        if tk.messagebox.askyesnocancel(title="Eliminar registro", message="¿Esta seguro de eliminar el registro '{}'?".format(registry.name), parent=self):
            FrameTakePhoto.delete_photo(patient_id=registry._id)
            self.primary_connection_db.collection_notification_delete(collection_name="Patient", _id=registry._id)
            self.root.notify_build_db_component(collection_name="Patient", registry=registry, op="delete", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges", "FrameScheduleCleaning", "FrameAuthenticatedPatientTable", "FrameCalendar", "FrameCalendarSpecificDate"])

    def button_authentication(self, registry):
        self.root.get_child(name="FrameAuthenticatedPatientTable").add_authenticated_patient(registry=registry)

    def button_create_new_registry(self):
        self.get_element(cad_pos="2,0").set_label(value="Registro nuevo en proceso...")
        self.show_element(cad_pos="3,0")
        self.get_element(cad_pos="3,0").set_registry(registry=Patient.get_empty())

    def button_save_registry(self):
        if self.get_element(cad_pos="3,0").is_visible:
            new_registry=self.get_element(cad_pos="3,0").get_registry()
            old_registry=self.get_element(cad_pos="3,0").registry
            if self.registry_verification(registry=new_registry):
                pathology_registries=self.get_element(cad_pos="3,0").get_element(cad_pos="9,0").registries
                disease_registries=self.get_element(cad_pos="3,0").get_element(cad_pos="11,0").registries
                new_registry.pathologies=self.update_and_get_attr(collection_name="Pathology", collection_class=Pathology, attr_registries=pathology_registries, old_registry=old_registry, new_registry=new_registry)
                new_registry.diseases=self.update_and_get_attr(collection_name="Disease", collection_class=Disease, attr_registries=disease_registries, old_registry=old_registry, new_registry=new_registry)
                
                # Guardamos foto (si es que se tomo foto)
                if self.get_element(cad_pos="3,0").get_element(cad_pos="1,0").save_photo(patient_id=new_registry._id):
                    # Notificamos cambios
                    self.root.notify_listener_db_component(collection_name="Patient", registry=new_registry, op="update", notify_listener_db_component_names=[self.name, "FrameAuthenticatedPatientTable", "FrameCalendarSpecificDate"])
                else:
                    FrameTakePhoto.delete_photo(patient_id=new_registry._id)
                    # Notificamos cambios
                    self.root.notify_listener_db_component(collection_name="Patient", registry=new_registry, op="update", notify_listener_db_component_names=[self.name, "FrameAuthenticatedPatientTable", "FrameCalendarSpecificDate"])

                if old_registry.name != "":
                    # Se actualizo un registro existente
                    obj_update=Patient.get_differences(old_obj=old_registry, new_obj=new_registry)
                    if len(obj_update["update"]) > 0:
                        self.primary_connection_db.collection_notification_update(collection_name="Patient", obj_update=obj_update)
                        self.root.notify_build_db_component(collection_name="Patient", registry=new_registry, op="update", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges", "FrameScheduleCleaning", "FrameAuthenticatedPatientTable", "FrameCalendar", "FrameCalendarSpecificDate"], old_registry=old_registry)
                    else:
                        self.button_close()
                else:
                    # Es un registro nuevo
                    self.primary_connection_db.collection_notification_insert(collection_name="Patient", obj=new_registry)
                    self.root.notify_build_db_component(collection_name="Patient", registry=new_registry, op="insert", notify_build_db_component_names=[self.name, "FrameNotifyDBChanges", "FrameScheduleCleaning", "FrameAuthenticatedPatientTable", "FrameCalendar", "FrameCalendarSpecificDate"])
        
                self.root.get_child(name="FrameCalendar").restart_calendar()
        else:
            tk.messagebox.showinfo(title="Guardar registro", message="No hay nada para guardar.", parent=self)

    def button_close(self):
        self.get_element(cad_pos="2,0").clear_label()
        self.hide_element(cad_pos="3,0")

    def registry_verification(self, registry):
        if registry is not None:
            if registry.name != "":
                return True
            else:
                tk.messagebox.showinfo(title="Registro no valido", message="El nombre no debe estar vacio.", parent=self)
                return False
        return False

    def update_and_get_attr(self, collection_name: Literal["Pathology", "Disease"], collection_class, attr_registries, old_registry, new_registry):
        temp_patient_attrs=[]
        # En comun
        intersection=set(new_registry.pathologies).intersection(old_registry.pathologies) if collection_name == "Pathology" else set(new_registry.diseases).intersection(old_registry.diseases)
        for _id in intersection:
            new_attr=list(filter(lambda elem: elem._id == _id, attr_registries))[0]
            old_attr=list(filter(lambda elem: elem._id == _id, self.primary_connection_db.db.collections[collection_name]))[0]
            if new_attr.name.lower() != old_attr.name.lower():
                # Creamos una nueva 
                temp_attr=collection_class(_id=ObjectId(), name=new_attr.name)
                temp_patient_attrs.append(temp_attr._id)
                self.primary_connection_db.collection_notification_insert(collection_name=collection_name, obj=temp_attr)
                self.root.notify_build_db_component(collection_name=collection_name, registry=temp_attr, op="insert", notify_build_db_component_names=["FrameNotifyDBChanges"])
            else:
                temp_patient_attrs.append(new_attr._id)
        # Nuevas 
        difference=set(new_registry.pathologies).difference(intersection) if collection_name == "Pathology" else set(new_registry.diseases).difference(intersection)
        for _id in difference:
            new_attr=list(filter(lambda elem: elem._id == _id, attr_registries))[0]
            if new_attr.name.lower() in list(map(lambda elem: elem.name.lower(), self.primary_connection_db.db.collections[collection_name])):
                # Usamos una ya existente
                temp_patient_attrs.append(list(filter(lambda elem: elem.name.lower() == new_attr.name.lower(), self.primary_connection_db.db.collections[collection_name]))[0]._id)
            else:
                # Creamos una nueva 
                temp_patient_attrs.append(new_attr._id)
                self.primary_connection_db.collection_notification_insert(collection_name=collection_name, obj=new_attr)
                self.root.notify_build_db_component(collection_name=collection_name, registry=new_attr, op="insert", notify_build_db_component_names=["FrameNotifyDBChanges"])
        return temp_patient_attrs

class FramePatient(CreateScrollableFrame, TemplateFrameClasses):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(14,1), arr=None), **kwargs)
        TemplateFrameClasses.__init__(self)
        self.registry: Patient=None
        self.var_patient_name=ctk.StringVar(value="")
        self.var_session_name=ctk.StringVar(value="")
        self.var_current_situation_name=ctk.StringVar(value="")
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="Foto", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameTakePhoto(master=self), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_label(master=self, text="Nombre", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_entry(master=self, textvariable=self.var_patient_name, justify=ctk.LEFT, width=250), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_label(master=self, text="Sesion", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="5,0", element=ctk.CTkOptionMenu(master=self, variable=self.var_session_name, values=[]), padx=1, pady=1)
        self.insert_element(cad_pos="6,0", element=FrameHelp.create_label(master=self, text="Situacion actual", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="7,0", element=ctk.CTkOptionMenu(master=self, variable=self.var_current_situation_name, values=[]), padx=1, pady=1)
        self.insert_element(cad_pos="8,0", element=FrameHelp.create_label(master=self, text="Patologias", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="9,0", element=FramePathologyBasicOperations(master=self, frame_search_width=200, table_width=400, table_height=300, sub_frame_width=300, sub_frame_height=200, name="FramePatientPathologyBasicOperations"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="10,0", element=FrameHelp.create_label(master=self, text="Enfermedades", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="11,0", element=FrameDiseaseBasicOperations(master=self, frame_search_width=200, table_width=400, table_height=300, sub_frame_width=300, sub_frame_height=200, name="FramePatientDiseaseBasicOperations"), padx=5, pady=5, sticky="") 
        self.insert_element(cad_pos="12,0", element=FrameHelp.create_label(master=self, text="Agenda", weight="bold", size=18, fg_color="coral"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="13,0", element=FrameScheduleBasicOperations(master=self, frame_search_width=200, table_width=400, table_height=300, sub_frame_width=300, sub_frame_height=400, name="FrameePatientScheduleBasicOperations"), padx=5, pady=5, sticky="")

    def clear_all(self):
        self.registry=None
        self.var_patient_name.set(value="")
        self.var_session_name.set(value="")
        self.var_current_situation_name.set(value="")
        self.get_element(cad_pos="5,0").configure(values=[])
        self.get_element(cad_pos="7,0").configure(values=[])
        self.get_element(cad_pos="9,0").button_close()
        self.get_element(cad_pos="11,0").button_close()
        self.get_element(cad_pos="13,0").button_close()

    def set_registry(self, registry: Patient):
        self.clear_all()
        db=self.primary_connection_db.db
        self.registry=registry

        patient_name=self.registry.name
        patient_session=list(filter(lambda elem: elem.name == "No especificado", db.collections["Session"]))[0] if self.registry.session is None else db.find_by_id(collection_name="Session", _id=self.registry.session)
        patient_pathologies=[db.find_by_id(collection_name="Pathology", _id=self.registry.pathologies[i]) for i in range(len(self.registry.pathologies))]
        patient_schedule=self.registry.schedule
        patient_diseases=[db.find_by_id(collection_name="Disease", _id=self.registry.diseases[i]) for i in range(len(self.registry.diseases))]
        patient_current_situation=list(filter(lambda elem: elem.name == "Activo", db.collections["CurrentSituation"]))[0] if self.registry.current_situation is None else db.find_by_id(collection_name="CurrentSituation", _id=self.registry.current_situation)
        session_list=db.collections["Session"]
        current_situation_list=db.collections["CurrentSituation"]

        self.var_patient_name.set(value=patient_name)
        self.var_session_name.set(value=patient_session.name)
        self.var_current_situation_name.set(value=patient_current_situation.name)
        self.get_element(cad_pos="1,0").set_image(patient_id=self.registry._id)
        self.get_element(cad_pos="5,0").configure(values=list(map(lambda elem: elem.name, session_list)))
        self.get_element(cad_pos="7,0").configure(values=list(map(lambda elem: elem.name, current_situation_list)))
        self.get_element(cad_pos="9,0").build_table(registries=copy.deepcopy(patient_pathologies), _ids=list(map(lambda elem: str(elem._id), patient_pathologies)))
        pathology_autocomplete_list=list(map(lambda elem: elem.name, db.collections["Pathology"]))
        self.get_element(cad_pos="9,0").get_element(cad_pos="1,1").get_element(cad_pos="0,0").set_autocomplete_list(pathology_autocomplete_list)
        self.get_element(cad_pos="11,0").build_table(registries=copy.deepcopy(patient_diseases), _ids=list(map(lambda elem: str(elem._id), patient_diseases)))
        disease_autocomplete_list=list(map(lambda elem: elem.name, db.collections["Disease"]))
        self.get_element(cad_pos="11,0").get_element(cad_pos="1,1").get_element(cad_pos="0,0").set_autocomplete_list(disease_autocomplete_list)
        self.get_element(cad_pos="13,0").build_table(registries=copy.deepcopy(patient_schedule), _ids=[ObjectId() for i in range(len(patient_schedule))])

    def get_registry(self):
        if self.registry is not None:
            db=self.primary_connection_db.db
            patient_session=list(filter(lambda elem: elem.name == self.var_session_name.get(), db.collections["Session"]))[0]._id
            patient_current_situation=list(filter(lambda elem: elem.name == self.var_current_situation_name.get(), db.collections["CurrentSituation"]))[0]._id
            patient_pathologies=self.get_element(cad_pos="9,0").registries
            patient_diseases=self.get_element(cad_pos="11,0").registries
            patient_schedule=self.get_element(cad_pos="13,0").registries
            return Patient(_id=self.registry._id, name=self.var_patient_name.get(), session=patient_session, pathologies=list(map(lambda elem: elem._id, patient_pathologies)), schedule=patient_schedule, diseases=list(map(lambda elem: elem._id, patient_diseases)), current_situation=patient_current_situation)
        return None

class FrameCalendar(CreateFrame, BuildComponent, BuildDBComponent):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.insert_element(cad_pos="0,0", element=FrameCalendarDate(master=self, calendar_text_size=12, calendar_visualization_text_size=16, calendar_width=300, calendar_height=300, calendar_callback=self.calendar_callback, name="FrameCalendarDate"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameCalendarSpecificDate(master=self, frame_search_width=200, table_width=800, table_height=400, name="FrameCalendarSpecificDate"), padx=5, pady=5, sticky="").hide_frame()
        self.insert_element(cad_pos="2,0", element=FrameNotCalendarSpecificDate(master=self), padx=5, pady=5, sticky="").hide_frame()

        self.restart_calendar()

    def build_component(self):
        self.get_element(cad_pos="0,0").clear_cal_event_dates()
        patient_list=self.primary_connection_db.db.collections["Patient"]
        for patient in patient_list:
            self.get_element(cad_pos="0,0").add_cal_event_dates(cal_event_dates=copy.deepcopy(patient.schedule), text=[patient.name for i in range(len(patient.schedule))])

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Patient":
            # Actualizamos el calendario
            if op == "insert": 
                self.get_element(cad_pos="0,0").add_cal_event_dates(cal_event_dates=registry.schedule, text=[registry.name for i in range(len(registry.schedule))])
            elif op == "update": 
                old_registry=kwargs["old_registry"]
                old_schedule=old_registry.schedule
                new_schedule=registry.schedule
                intersection=set(new_schedule).intersection(set(old_schedule))
                # Lo nuevo
                difference=list(set(new_schedule).difference(intersection))
                self.get_element(cad_pos="0,0").add_cal_event_dates(cal_event_dates=difference, text=[registry.name for i in range(len(difference))])
                # Lo eliminado
                difference=list(set(old_schedule).difference(set(new_schedule)))
                self.get_element(cad_pos="0,0").delete_cal_event_by_dates(cal_event_dates=difference)
            elif op == "delete": 
                self.get_element(cad_pos="0,0").delete_cal_event_by_dates(cal_event_dates=registry.schedule)

    def calendar_callback(self, date):
        self.get_element(cad_pos="1,0").set_table(date=date)
        if len(self.get_element(cad_pos="1,0").registries) == 0:
            self.hide_element(cad_pos="1,0")
            self.show_element(cad_pos="2,0")
        else:
            self.show_element(cad_pos="1,0")
            self.hide_element(cad_pos="2,0")

    def restart_calendar(self):
        self.build_component()
        self.get_element(cad_pos="0,0").calendar_trace_callback(None, None, None)

class FrameNotCalendarSpecificDate(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="   No hay citas para la fecha seleccionada", fg_color="lightcoral", weight="bold", size=16, img_name="not_search", img_width=50, img_height=50), padx=5, pady=5, sticky="")

class FrameCalendarSpecificDate(FrameTableDB, BuildComponent, BuildDBComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db
        self.date=None
        self.session_list=self.primary_connection_db.db.collections["Session"]
        self.session_names=list(map(lambda elem: elem.name, self.session_list))
        self.session_ids=list(map(lambda elem: elem._id, self.session_list))

        self.build_component()

    def build_component(self):
        pass

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Patient" and self.is_visible:
            # Actualizamos la tabla de pacientes especificos
            registry_ids=list(map(lambda elem: elem._id, self.registries)) 
            if op == "insert":
                if self.date in list(map(lambda elem: elem.split(" ")[0], registry.schedule)):
                    self.insert_registry(registry=registry, _id=str(registry._id))
            elif op == "update":
                if registry._id in registry_ids:
                    if self.date in list(map(lambda elem: elem.split(" ")[0], registry.schedule)):
                        self.update_registry(registry=registry, _id=str(registry._id))
                    else:
                        self.delete_registry(_id=str(registry._id))
            elif op == "delete":
                if registry._id in registry_ids:
                    self.delete_registry(_id=str(registry._id))

    def listener_db_component(self, collection_name, registry, **kwargs):
        if collection_name == "Patient" and self.is_visible:
            self.update_registry(registry=registry, _id=str(registry._id))

    def update_table_item(self, table_item, old_registry, new_registry):
        image=ui_images.get_patient_photo(patient_id=new_registry._id, width=100, height=100)
        table_item.get_element(cad_pos="0,1").configure(image=image)
        table_item.get_element(cad_pos="0,1").image=image
        if new_registry.name != old_registry.name:
            table_item.get_element(cad_pos="0,2").get_element(cad_pos="1,0").configure(text=new_registry.name)

        specific_schedule=list(filter(lambda date_hour: self.date == date_hour.split(" ")[0], new_registry.schedule))
        specific_schedule.sort()
        specific_hour=list(map(lambda elem: elem.split(" ")[1], specific_schedule))
        table_item.destroy_element(cad_pos="0,3")
        table_item.insert_element(cad_pos="0,3", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Citas", l=specific_hour, width=300, height=100), padx=5, pady=5, sticky="")

        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="1,0").configure(command=lambda: self.button_authentication(registry=new_registry))

    def create_table_item(self, master, registry):
        specific_schedule=list(filter(lambda date_hour: self.date == date_hour.split(" ")[0], registry.schedule))
        specific_schedule.sort()
        specific_hour=list(map(lambda elem: elem.split(" ")[1], specific_schedule))

        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(1,4), arr=None), height=200, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(2,1), arr=None), width=70, height=150)
        buttons_table_item.enable_fixed_size()
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_authentication(registry=registry), fg_color="transparent", img_name="authentication", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_photo_item(master=table_item, patient_id=registry._id, photo_width=150, photo_height=150), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,2", element=FrameHelp.create_table_sub_item(master=table_item, title="Nombre", text=registry.name, fg_color="transparent", width=200, height=80), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,3", element=FrameHelp.create_table_list_sub_item(master=table_item, title="Citas", l=specific_hour, width=300, height=100), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def button_select(self, registry):
        self.root.get_child(name="FramePatientBasicOperations").button_select(registry=registry)

    def button_authentication(self, registry):
        self.root.get_child(name="FrameAuthenticatedPatientTable").add_authenticated_patient(registry=registry)

    def search_compare(self, registry):
        return registry.name
            
    def set_table(self, date):
        self.date=date
        filtered_registries=list(filter(lambda elem: date in list(map(lambda date_hour: date_hour.split(" ")[0], elem.schedule)), self.primary_connection_db.db.collections["Patient"]))
        filtered_ids=list(map(lambda elem: str(elem._id), filtered_registries))
        self.build_table(registries=copy.deepcopy(filtered_registries), _ids=filtered_ids)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class FrameRight(CreateScrollableFrame, BuildComponent):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        BuildComponent.__init__(self)
        
        self.build_component()

    def build_component(self):
        self.destroy_all()
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="", fg_color="transparent", img_name="user_authentication", img_width=100, img_height=100), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameAuthenticatedPatientTable(master=self, frame_search_width=200, table_width=400, table_height=600, name="FrameAuthenticatedPatientTable"), padx=5, pady=5, sticky="nsew")

class FrameAuthenticatedPatientTable(FrameTableDB, BuildComponent, BuildDBComponent):
    def __init__(self, master, frame_search_width, table_width, table_height, **kwargs):
        FrameTableDB.__init__(self, master=master, registries=None, _ids=None, frame_search_width=frame_search_width, table_width=table_width, table_height=table_height, **kwargs)
        BuildComponent.__init__(self)
        BuildDBComponent.__init__(self)
        self.root=self.get_root()
        self.primary_connection_db: ConnectionDB=self.root.primary_connection_db

        self.build_component()

    def build_component(self):
        pass

    def build_db_component(self, collection_name, registry, op: Literal['insert', 'update', 'delete'], **kwargs):
        if collection_name == "Patient":
            # Actualizamos la tabla
            if op == "update": self.update_registry(registry=registry, _id=str(registry._id))
            elif op == "delete": self.delete_registry(_id=str(registry._id))

    def listener_db_component(self, collection_name, registry, **kwargs):
        if collection_name == "Patient":
            self.update_registry(registry=registry, _id=str(registry._id))

    def update_table_item(self, table_item, old_registry, new_registry):
        image=ui_images.get_patient_photo(patient_id=new_registry._id, width=100, height=100)
        table_item.get_element(cad_pos="1,0").configure(image=image)
        table_item.get_element(cad_pos="1,0").image=image
        if new_registry.name != old_registry.name:
            table_item.get_element(cad_pos="1,1").get_element(cad_pos="1,0").configure(text=new_registry.name)
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,0").configure(command=lambda: self.button_select(registry=new_registry))
        table_item.get_element(cad_pos="0,0").get_element(cad_pos="0,1").configure(command=lambda: self.button_delete(registry=new_registry))
        
    def create_table_item(self, master, registry):
        table_item_container=CreateFrame(master=master)
        table_item=CreateScrollableFrame(master=table_item_container, grid_frame=GridFrame(dim=(2,2), arr=np.array([["0,0","0,0"],["1,0","1,1"]])), height=200, fg_color="lightcoral", orientation="horizontal")
        buttons_table_item=CreateFrame(master=table_item, grid_frame=GridFrame(dim=(1,2), arr=None), height=60)
        buttons_table_item.insert_element(cad_pos="0,0", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_select(registry=registry), fg_color="transparent", img_name="select", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        buttons_table_item.insert_element(cad_pos="0,1", element=FrameHelp.create_button(master=buttons_table_item, command=lambda: self.button_delete(registry=registry), fg_color="transparent", img_name="session_ended", img_width=30, img_height=30), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="0,0", element=buttons_table_item, padx=5, pady=5, sticky="ew")
        table_item.insert_element(cad_pos="1,0", element=FrameHelp.create_photo_item(master=table_item, patient_id=registry._id, photo_width=100, photo_height=100), padx=5, pady=5, sticky="")
        table_item.insert_element(cad_pos="1,1", element=FrameHelp.create_table_sub_item(master=table_item, title="Nombre", text=registry.name, fg_color="transparent", width=250, height=80), padx=5, pady=5, sticky="")
        table_item_container.insert_element(cad_pos="0,0", element=table_item, padx=0, pady=0, sticky="nsew")
        return table_item_container

    def search_compare(self, registry):
        return registry.name
    
    def button_select(self, registry):
        self.root.get_child(name="FramePatientBasicOperations").button_select(registry=registry)

    def button_delete(self, registry):
        if tk.messagebox.askyesnocancel(title="Finalizar sesion", message="¿Esta seguro de finalizar la sesion para '{}'?".format(registry.name), parent=self):
            self.delete_registry(_id=str(registry._id))

    def add_authenticated_patient(self, registry):
        if str(registry._id) not in self._ids:
            self.insert_registry(registry=registry, _id=str(registry._id))
        else:
            tk.messagebox.showinfo(title="Autenticacion", message="El paciente '{}' ya ha sido autenticado.".format(registry.name), parent=self)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class RootDBNotification(CreateFrame):
    def __init__(self, **kwargs):
        CreateFrame.__init__(self, **kwargs)

    def notify_build_db_component(self, collection_name, registry, op, notify_build_db_component_names, elem=None, **kwargs):
        if elem is None:
            elem=self
        for cad_pos in elem.elements.keys():
            if elem.element_exists(cad_pos=cad_pos):
                child=elem.get_element(cad_pos=cad_pos)
                is_leaf=elem.is_leaf(cad_pos=cad_pos)
                if child is not None and not is_leaf:
                    if hasattr(child, 'build_db_component'):
                        if notify_build_db_component_names is None:
                            child.build_db_component(collection_name=collection_name, registry=registry, op=op, **kwargs)
                        elif child.name in notify_build_db_component_names:
                            child.build_db_component(collection_name=collection_name, registry=registry, op=op, **kwargs)
                    self.notify_build_db_component(collection_name=collection_name, registry=registry, op=op, notify_build_db_component_names=notify_build_db_component_names, elem=child, **kwargs)

    def notify_listener_db_component(self, collection_name, registry, notify_listener_db_component_names, elem=None, **kwargs):
        if elem is None:
            elem=self
        for cad_pos in elem.elements.keys():
            if elem.element_exists(cad_pos=cad_pos):
                child=elem.get_element(cad_pos=cad_pos)
                is_leaf=elem.is_leaf(cad_pos=cad_pos)
                if child is not None and not is_leaf:
                    if hasattr(child, 'listener_db_component'):
                        if notify_listener_db_component_names is None:
                            child.listener_db_component(collection_name=collection_name, registry=registry, **kwargs)
                        elif child.name in notify_listener_db_component_names:
                            child.listener_db_component(collection_name=collection_name, registry=registry, **kwargs)
                    self.notify_listener_db_component(collection_name=collection_name, registry=registry, notify_listener_db_component_names=notify_listener_db_component_names, elem=child, **kwargs)

class FrameDatabaseApplication(RootDBNotification, BuildComponent):
    def __init__(self, master, primary_connection_db: ConnectionDB, thread_camera_1, thread_camera_2, **kwargs):
        RootDBNotification.__init__(self, master=master, grid_frame=GridFrame(dim=(1,10), arr=np.array([["0,0","0,0","0,0","0,0","0,0","0,0","0,0","0,0","0,8","0,8"]])), **kwargs)
        BuildComponent.__init__(self)
        self.primary_connection_db=primary_connection_db
        self.thread_camera_1=thread_camera_1
        self.thread_camera_2=thread_camera_2

        self.build_component()

    def build_component(self):
        self.destroy_all()
        self.insert_element(cad_pos="0,0", element=FrameLeft(master=self, name="FrameLeft"), padx=5, pady=5, sticky="nsew")
        self.insert_element(cad_pos="0,8", element=FrameRight(master=self, name="FrameRight"), padx=5, pady=5, sticky="nsew")

        

