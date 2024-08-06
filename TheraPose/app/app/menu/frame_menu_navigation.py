import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.frame_help import FrameHelp

class FrameMenuNavigation(CreateFrame):
    def __init__(self, master, **kwargs):
        CreateFrame.__init__(self, master=master, grid_frame=GridFrame(dim=(5,1), arr=None), **kwargs) 
        self.root=self.get_root()
        self.selected_page="home"

        self.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=self, text="  TheraPose", weight="bold", size=14, img_name="logo", img_width=50, img_height=50, fg_color="lightcoral", compound="left"), padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=FrameHelp.create_button(master=self, img_name="home", corner_radius=0, height=40, border_spacing=10, text="Inicio", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=lambda: self.button_selected_page(selected_page="home")), padx=5, pady=0, sticky="ew")
        self.insert_element(cad_pos="2,0", element=FrameHelp.create_button(master=self, img_name="calibration", corner_radius=0, height=40, border_spacing=10, text="Calibracion", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=lambda: self.button_selected_page(selected_page="calibration")), padx=5, pady=0, sticky="ew")
        self.insert_element(cad_pos="3,0", element=FrameHelp.create_button(master=self, img_name="estimation", corner_radius=0, height=40, border_spacing=10, text="Estimacion", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=lambda: self.button_selected_page(selected_page="estimation")), padx=5, pady=0, sticky="ew")        
        self.insert_element(cad_pos="4,0", element=FrameHelp.create_button(master=self, img_name="game", corner_radius=0, height=40, border_spacing=10, text="Juegos", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=lambda: self.button_selected_page(selected_page="game")), padx=5, pady=0, sticky="ew")        
        
        self.button_selected_page(selected_page=self.selected_page)

    def button_selected_page(self, selected_page):
        self.selected_page=selected_page
        
        self.get_element(cad_pos="1,0").configure(fg_color=("gray75", "gray25") if selected_page == "home" else "transparent")
        self.get_element(cad_pos="2,0").configure(fg_color=("gray75", "gray25") if selected_page == "calibration" else "transparent")
        self.get_element(cad_pos="3,0").configure(fg_color=("gray75", "gray25") if selected_page == "estimation" else "transparent")
        self.get_element(cad_pos="4,0").configure(fg_color=("gray75", "gray25") if selected_page == "game" else "transparent")

        self.root.select_page(page_name=self.selected_page)

 