import customtkinter  as ctk

from general.grid_frame import GridFrame
from general.create_frame import CreateFrame
from general.create_scrollable_frame import CreateScrollableFrame
import general.ui_images as ui_images

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

class FrameHelp:

    @staticmethod
    def get_some_image(img_type: Literal["image", "algorithm_image", "collection_image", "patient_photo", "game_image", "image_to_canvas"], img_name, img_width, img_height):
        if img_type == "image":
            return ui_images.get_image(name=img_name, width=img_width, height=img_height)
        elif img_type == "algorithm_image":
            return ui_images.get_algorithm_image(name=img_name, width=img_width, height=img_height)
        elif img_type == "collection_image":
            return ui_images.get_collection_image(name=img_name, width=img_width, height=img_height)
        elif img_type == "patient_photo":
            return ui_images.get_patient_photo(patient_id=img_name, width=img_width, height=img_height)
        elif img_type == "game_image":
            return ui_images.get_game_image(name=img_name, width=img_width, height=img_height)
        elif img_type == "image_to_canvas":
            return ui_images.get_image_to_canvas(name=img_name, width=img_width, height=img_height)

    @staticmethod
    def create_photo_item(master, patient_id, photo_width=150, photo_height=150):
        return ctk.CTkLabel(master=master, text="", image=ui_images.get_patient_photo(patient_id=patient_id, width=photo_width, height=photo_height))
        
    @staticmethod
    def create_table_list_sub_item(master, title, l, width=100, height=100, **kwargs):
        table_list_sub_item=CreateFrame(master=master, grid_frame=GridFrame(dim=(1,2), arr=None), width=width, height=height, **kwargs)
        frame_list=CreateScrollableFrame(master=table_list_sub_item, grid_frame=GridFrame(dim=(len(l), 1), arr=None), orientation="vertical")
        for i in range(len(l)):
            list_item=l[i]
            frame_list.insert_element(cad_pos="{},0".format(i), element=FrameHelp.create_label(master=frame_list, text=list_item, family="Modern", size=12, weight="normal"), padx=5, pady=5, sticky="ew")
        table_list_sub_item.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=table_list_sub_item, text=title, family="Modern", size=12, weight="bold"), padx=5, pady=5, sticky="ew")
        table_list_sub_item.insert_element(cad_pos="0,1", element=frame_list, padx=5, pady=5, sticky="ew")
        table_list_sub_item.enable_fixed_size()
        return table_list_sub_item
    
    @staticmethod
    def create_table_sub_item(master, title, text, size=12, width=100, height=100, **kwargs):
        table_sub_item=CreateFrame(master=master, grid_frame=GridFrame(dim=(2,1), arr=None), width=width, height=height, **kwargs)
        table_sub_item.insert_element(cad_pos="0,0", element=FrameHelp.create_label(master=table_sub_item, text=title, family="Modern", size=size, weight="bold"), padx=5, pady=5, sticky="ew")
        table_sub_item.insert_element(cad_pos="1,0", element=FrameHelp.create_label(master=table_sub_item, text=text, family="Modern", size=size, weight="normal"), padx=5, pady=5, sticky="ew")
        table_sub_item.enable_fixed_size()
        return table_sub_item

    @staticmethod
    def create_label(master, corner_radius=10, text="", family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", fg_color: Literal["white", "transparent", "coral", "greenyellow", "lightcoral", "lightpink", "pink"]="white", img_type: Literal["image", "algorithm_image", "collection_image", "patient_photo", "game_image"]="image", img_name="", img_width=30, img_height=30, compound=ctk.LEFT, **kwargs):
        font_label=ctk.CTkFont(family=family, size=size, weight=weight)
        image=FrameHelp.get_some_image(img_type=img_type, img_name=img_name, img_width=img_width, img_height=img_height)
        return ctk.CTkLabel(master=master, text=text, font=font_label, fg_color=fg_color, corner_radius=corner_radius, image=image, compound=compound, **kwargs)
    
    @staticmethod
    def create_button(master, corner_radius=10, text="", command=None, family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", fg_color: Literal["green3", "transparent", "green", "greenyellow", "coral", "white", "lightcoral", "lightpink", "pink"]="green3", hover_color: Literal["gray60"]="gray60", img_type: Literal["image", "algorithm_image", "collection_image", "patient_photo", "game_image"]="image", img_name="", img_width=30, img_height=30, compound=ctk.TOP, **kwargs):
        font_button=ctk.CTkFont(family=family, size=size, weight=weight)
        image=FrameHelp.get_some_image(img_type=img_type, img_name=img_name, img_width=img_width, img_height=img_height)
        return ctk.CTkButton(master=master, text=text, font=font_button, corner_radius=corner_radius, command=command, fg_color=fg_color, hover_color=hover_color, image=image, compound=compound, **kwargs)

    @staticmethod
    def create_entry(master, family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", **kwargs):
        font_entry=ctk.CTkFont(family=family, size=size, weight=weight)
        return ctk.CTkEntry(master=master, font=font_entry, **kwargs)
    
    @staticmethod
    def create_radio_button(master, family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", **kwargs):
        font_radio_button=ctk.CTkFont(family=family, size=size, weight=weight)
        return ctk.CTkRadioButton(master=master, font=font_radio_button, **kwargs)
    
    @staticmethod
    def create_check_box(master, family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", **kwargs):
        font_check_box=ctk.CTkFont(family=family, size=size, weight=weight)
        return ctk.CTkCheckBox(master=master, font=font_check_box, **kwargs)
    
    @staticmethod
    def create_option_menu(master, family: Literal['Modern']="Modern", size=12, weight: Literal["normal", "bold"]="normal", **kwargs):
        font_option_menu=ctk.CTkFont(family=family, size=size, weight=weight)
        return ctk.CTkOptionMenu(master=master, font=font_option_menu, **kwargs)