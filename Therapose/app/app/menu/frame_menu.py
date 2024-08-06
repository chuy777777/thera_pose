from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame

from menu.frame_menu_navigation import FrameMenuNavigation
from menu.frame_select_camera import FrameSelectCamera

from connection_db.frame_menu_connection_db import FrameMenuConnectionDB

class FrameMenu(CreateScrollableFrame):
    def __init__(self, master, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs) 

        self.insert_element(cad_pos="1,0", element=FrameSelectCamera(master=self, fg_color='lightpink'), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="0,0", element=FrameMenuNavigation(master=self, fg_color='lightpink'), padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=FrameMenuConnectionDB(master=self, fg_color='lightpink'), padx=5, pady=5, sticky="ew")