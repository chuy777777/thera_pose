import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp

class FrameProgressBar(CreateFrame):
    def __init__(self, master, title, width, height, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs) 
        
        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text=title, size=14, weight="bold", fg_color="coral"), padx=5, pady=5)
        self.insert_element(cad_pos="1,0", element=ctk.CTkProgressBar(master=self, width=width, height=height, orientation="horizontal", mode="indeterminate"), padx=5, pady=5)

    def start_progress_bar(self):
        self.show_frame()
        self.get_element(cad_pos="1,0").start()

    def stop_progress_bar(self):
        self.hide_frame()
        self.get_element(cad_pos="1,0").stop()