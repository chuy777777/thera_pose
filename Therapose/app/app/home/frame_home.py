import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp

class FrameHome(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="", img_name="home_page", img_width=800, img_height=600), padx=20, pady=20, sticky='')