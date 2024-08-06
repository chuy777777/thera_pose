import customtkinter  as ctk
import os
from PIL import Image

global_path_app=os.path.dirname(os.path.abspath('__file__'))
global_path_app_images=os.path.join(global_path_app, "images")

def get_image(name, width, height):
    path=os.path.join(global_path_app_images, "{}.png".format(name))
    if os.path.exists(path):
        return ctk.CTkImage(Image.open(path), size=(width, height))
    else:
        return None

def get_algorithm_image(name, width, height):
    path=os.path.join(global_path_app_images, "algorithms/{}.png".format(name))
    if os.path.exists(path):
        return ctk.CTkImage(Image.open(path), size=(width, height))
    else:
        return None

def get_collection_image(name, width, height):
    path=os.path.join(global_path_app_images, "collections/{}.png".format(name))
    if os.path.exists(path):
        return ctk.CTkImage(Image.open(path), size=(width, height))
    else:
        return None
    
def get_patient_photo(patient_id, width, height):
    path=os.path.join(global_path_app_images, "patients/{}/{}.png".format(patient_id, patient_id))
    if os.path.exists(path):
        return ctk.CTkImage(Image.open(path), size=(width, height))
    else:
        path=os.path.join(global_path_app_images, "no_user.png")
        return ctk.CTkImage(Image.open(path), size=(width, height))

def get_game_image(name, width, height):
    path=os.path.join(global_path_app_images, "games/{}.png".format(name))
    if os.path.exists(path):
        return ctk.CTkImage(Image.open(path), size=(width, height))
    else:
        return None
    
def get_game_image_path(name):
    path=os.path.join(global_path_app_images, "games/{}.png".format(name))
    if os.path.exists(path):
        return path
    else:
        return None
    
def get_image_to_canvas(name, width, height):
    path=os.path.join(global_path_app_images, "games/{}.png".format(name))
    if os.path.exists(path):
        image=Image.open(path)
        image=image.resize(size=(width, height))
        return image
    else:
        path=os.path.join(global_path_app_images, "games/default.png")
        image=Image.open(path)
        image=image.resize(size=(width, height))
        return image
    
