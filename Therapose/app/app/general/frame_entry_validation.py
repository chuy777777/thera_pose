import numpy as np
import customtkinter  as ctk
import re

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp

class GroupClassValidation():
    def __init__(self, list_class_validation):
        self.list_class_validation=list_class_validation

    def is_valid(self):
        for class_validation in self.list_class_validation:
            if not class_validation.is_valid():
                return False
        return True
    
class TextValidation():
    def __init__(self, var_entry, max_length=0, regular_expression=""):
        self.var_entry=var_entry
        self.max_length=max_length
        self.regular_expression=regular_expression
        self.message=""
    
    def is_valid(self):
        if len(self.var_entry.get()) <= self.max_length:
            if self.regular_expression == "":
                self.message=""
                return True
            elif self.is_match(self.var_entry.get()):
                self.message=""
                return True
            else:
                self.message="El texto no coincide con la expresion regular dada: '{}'".format(self.regular_expression)
                return False
        else:
            self.message="El texto excede la longitud establecida (maximo {} caracteres)".format(self.max_length)
            return False
    
    def is_match(self, val):
        prog=re.compile(self.regular_expression)
        return False if prog.fullmatch(val) is None else True

class NumberValidation():
    def __init__(self, var_entry, min_val, max_val):
        self.var_entry=var_entry
        self.min_val=min_val
        self.max_val=max_val
        self.message=""

    def is_valid(self):
        if self.is_float(self.var_entry.get()) or self.is_int(self.var_entry.get()):
            val=float(self.var_entry.get())
            if val >= self.min_val and val <= self.max_val:
                self.message=""
                return True
            else: 
                self.message="El valor no se encuentra entre los limites establecidos (el valor debe estar entre [{}, {}])".format(self.min_val, self.max_val)
                return False
        else:
            self.message="Debes escribir un numero valido"
            return False

    def is_float(self, val):
        try:
            float(val)
            return True
        except ValueError:
            return False
    
    def is_int(self, val):
        return val.isnumeric()

class FrameEntryValidation(CreateFrame):
    def __init__(self, master, class_validation, text_title, text_show="", entry_width=250, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        self.class_validation=class_validation
        self.var_message=ctk.StringVar(value="")

        self.class_validation.var_entry.trace_add("write", self.callback_trace)
        
        self.frame_message=CreateFrame(master=self)
        self.frame_message.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self.frame_message, textvariable=self.var_message, text="", text_color="red", wraplength=200), padx=5, pady=5)

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text=text_title), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_entry(master=self, show=text_show, width=entry_width, textvariable=self.class_validation.var_entry, justify='center'), padx=5, pady=5, sticky="nsew")
        self.insert_element(cad_pos="2,0", element=self.frame_message, padx=5, pady=5, sticky="nsew").hide_frame()

    def callback_trace(self, var, index, mode):
        if not self.class_validation.is_valid():
            self.var_message.set(value=self.class_validation.message)
            self.frame_message.show_frame()
        else:
            self.var_message.set(value="")
            self.frame_message.hide_frame()

    


