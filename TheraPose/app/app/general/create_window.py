import customtkinter  as ctk

from general.create_scrollable_frame import CreateScrollableFrame
from general.create_frame import CreateFrame
from general.grid_frame import GridFrame

class CreateWindow(ctk.CTkToplevel):
    def __init__(self, window_title, window_geometry, scrollable=False, on_closing_callback=None, grid_frame_root=GridFrame(), **kwargs):
        ctk.CTkToplevel.__init__(self)
        self.window_title=window_title
        self.window_geometry=window_geometry 
        self.scrollable=scrollable
        self.on_closing_callback=on_closing_callback
        self.grid_frame_root=grid_frame_root

        # Configuramos nuestra aplicacion
        if type(self.window_geometry) is tuple:
            # (width,heigh)
            self.geometry(str(self.window_geometry[0])+"x"+str(self.window_geometry[1]))
        elif self.window_geometry=="fullscreen":
            self.attributes('-fullscreen', True)
        self.title(self.window_title)
        
        # Configuramos el sistema de cuadricula
        self.grid_rowconfigure(0, weight=1)  
        self.grid_columnconfigure(0, weight=1)
        
        # Creamos un frame root
        self.frame_root=CreateScrollableFrame(master=self, grid_frame=self.grid_frame_root) if self.scrollable else CreateFrame(master=self, grid_frame=self.grid_frame_root)
        
       # Colocamos el frame root en la cuadricula
        self.frame_root.grid(row=0, column=0, **kwargs) # Al agregar sticky='nsew' el frame pasa de widthxheight a abarcar todo el espacio disponible
 
        # Creamos elementos y los insertamos a los frames
        # Los elementos se pueden ir insertando despues de haber creado la ventana

        # Configuramos el cerrado de la ventana
        self.protocol("WM_DELETE_WINDOW", self.close_window)
    
    def close_window(self):
        if not self.on_closing_callback is None:
            self.on_closing_callback()
        ctk.CTkToplevel.destroy(self)




