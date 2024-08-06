import customtkinter  as ctk
from bson import ObjectId

from general.grid_frame import GridFrame
from general.tamplate_frame import TemplateFrame

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

class CreateScrollableFrame(ctk.CTkScrollableFrame, TemplateFrame):
    def __init__(self, master, grid_frame=GridFrame(dim=(1,1), arr=None), is_root=False, name=str(ObjectId()), fg_color: Literal[None, "transparent", "white", "coral", "greenyellow", "lightcoral", "lightpink", "pink"]=None, **kwargs):
        ctk.CTkScrollableFrame.__init__(self, master=master, fg_color=fg_color, **kwargs)
        TemplateFrame.__init__(self, father=master, grid_frame=grid_frame, is_root=is_root, name=name, grid_information=self.grid_info())

        self.create_specific_grid_frame(grid_frame=grid_frame)

        # self.bind_all("<Button-4>", lambda e: self._parent_canvas.yview("scroll", -1, "units"))
        # self.bind_all("<Button-5>", lambda e: self._parent_canvas.yview("scroll", 1, "units"))

    def destroy(self):
        ctk.CTkScrollableFrame.destroy(self)

    def create_specific_grid_frame(self, grid_frame: GridFrame):
        self.grid_frame=grid_frame
        for cad_pos in list(self.elements.keys()):
            self.elements[cad_pos]["element"].destroy()
        self.elements={}
        
        h,w=self.grid_frame.dim
        for i in range(h):
            self.grid_rowconfigure(i, weight=1) 
            for j in range(w): 
                self.grid_columnconfigure(j, weight=1)

    def hide_frame(self):
        if self.is_visible:
            self.is_visible=False
            self.grid_information=self.grid_info()
            self.grid_forget()

    def show_frame(self):
        if not self.is_visible:
            self.is_visible=True
            self.grid(**self.grid_information)

    def toggle_visibility(self):
        if self.is_visible:
            self.hide_frame()
        else:
            self.show_frame()

# class CreateScrollableFrame(ctk.CTkScrollableFrame):
#     def __init__(self, master, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs):
#         ctk.CTkScrollableFrame.__init__(self, master=master, **kwargs)
#         self.grid_information=self.grid_info()
#         self.elements={}
#         self.is_visible=True

#         self.create_specific_grid_frame(grid_frame=grid_frame)

#     def destroy(self):
#         ctk.CTkScrollableFrame.destroy(self)

#     def create_specific_grid_frame(self, grid_frame: GridFrame):
#         self.grid_frame=grid_frame
#         for cad_pos in list(self.elements.keys()):
#             self.elements[cad_pos]["element"].destroy()
#         self.elements={}
        
#         h,w=self.grid_frame.dim
#         for i in range(h):
#             self.grid_rowconfigure(i, weight=1) 
#             for j in range(w): 
#                 self.grid_columnconfigure(j, weight=1)

#     def hide_frame(self):
#         if self.is_visible:
#             self.is_visible=False
#             self.grid_information=self.grid_info()
#             self.grid_forget()

#     def show_frame(self):
#         if not self.is_visible:
#             self.is_visible=True
#             self.grid(**self.grid_information)

#     def toggle_visibility(self):
#         if self.is_visible:
#             self.hide_frame()
#         else:
#             self.show_frame()

#     # Este metodo nos permite verificar si ha sido destruido el elemento o no
#     def element_exists(self, cad_pos):
#         return self.key_exists(cad_pos=cad_pos) and self.elements[cad_pos]["element"].winfo_exists()

#     # Este metodo nos permite verificar si una clave existe en el diccionario
#     def key_exists(self, cad_pos):
#         return cad_pos in list(self.elements.keys())

#     def insert_element(self, cad_pos, element, **kwargs):
#         i,j=[int(val) for val in cad_pos.split(",")]
#         columnspan=self.grid_frame.dict[cad_pos]["columnspan"]
#         rowspan=self.grid_frame.dict[cad_pos]["rowspan"]
#         element.grid(row=i, column=j, rowspan=rowspan, columnspan=columnspan, **kwargs)
#         self.elements[cad_pos]={"element": element, "kwargs": kwargs}
#         return element

#     def get_element(self,cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             return self.elements[cad_pos]["element"]
#         return None

#     def hide_element(self, cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             self.elements[cad_pos]["element"].grid_forget()

#     def show_element(self, cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             i,j=[int(val) for val in cad_pos.split(",")]
#             columnspan=self.grid_frame.dict[cad_pos]["columnspan"]
#             rowspan=self.grid_frame.dict[cad_pos]["rowspan"]
#             self.elements[cad_pos]["element"].grid(row=i, column=j, rowspan=rowspan, columnspan=columnspan, **self.elements[cad_pos]["kwargs"])

#     def destroy_element(self, cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             self.elements[cad_pos]["element"].destroy()
#             del self.elements[cad_pos]

#     # Solo utilizar cuando se trabaje con listas
#     def insert_element_at_end(self, element, **kwargs):
#         dim=self.grid_frame.dim
#         if dim[0] == 1 and len(self.elements) == 0:
#             # Significa que no hay ningun elemento todavia (el frame esta inicializado pero no tiene elementos)
#             element.grid(row=0, column=0, columnspan=dim[1], **kwargs)
#             self.elements["0,0"]={"element": element, "kwargs": kwargs}
#             return element
#         else:
#             element.grid(row=dim[0], column=0, columnspan=dim[1], **kwargs)
#             self.elements["{},0".format(dim[0])]={"element": element, "kwargs": kwargs}
#             aux=np.array([["{},0".format(dim[0])] for i in range(dim[1])], dtype="<U5").T
#             self.grid_frame=GridFrame(dim=(dim[0] + 1, dim[1]), arr=np.concatenate([self.grid_frame.arr, aux], axis=0))
#             return element
    
      