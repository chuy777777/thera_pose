import customtkinter  as ctk
from bson import ObjectId

from general.grid_frame import GridFrame
from general.tamplate_frame import TemplateFrame

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

class CreateFrame(ctk.CTkFrame, TemplateFrame):
    def __init__(self, master, grid_frame=GridFrame(dim=(1,1), arr=None), is_root=False, name=str(ObjectId()), fg_color: Literal[None, "transparent", "white", "coral", "greenyellow", "lightcoral", "lightpink", "pink"]=None, **kwargs):
        ctk.CTkFrame.__init__(self, master=master, fg_color=fg_color, **kwargs)
        TemplateFrame.__init__(self, father=master, grid_frame=grid_frame, is_root=is_root, name=name, grid_information=self.grid_info())
        
        self.create_specific_grid_frame(grid_frame=grid_frame)

    def destroy(self):
        ctk.CTkFrame.destroy(self)

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
    
    def enable_fixed_size(self):
        self.grid_propagate(False)

    def desable_fixed_size(self):
        self.grid_propagate(True)

# class CreateFrame(ctk.CTkFrame):
#     def __init__(self, master, grid_frame=GridFrame(dim=(1,1), arr=None), is_root=False, name=str(ObjectId()), fg_color: Literal[None, "transparent", "white", "coral", "greenyellow", "lightcoral", "lightpink", "pink"]=None, **kwargs):
#         ctk.CTkFrame.__init__(self, master=master, fg_color=fg_color, **kwargs)
#         self.father=master
#         self.is_root=is_root
#         self.name=name
#         self.child_search=None
#         self.grid_information=self.grid_info()
#         self.elements={}
#         self.is_visible=True

#         self.create_specific_grid_frame(grid_frame=grid_frame)

#     def destroy(self):
#         ctk.CTkFrame.destroy(self)

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
    
#     def enable_fixed_size(self):
#         self.grid_propagate(False)

#     def desable_fixed_size(self):
#         self.grid_propagate(True)

#     def insert_element(self, cad_pos, element, is_leaf=False, **kwargs):
#         i,j=[int(val) for val in cad_pos.split(",")]
#         columnspan=self.grid_frame.dict[cad_pos]["columnspan"]
#         rowspan=self.grid_frame.dict[cad_pos]["rowspan"]
#         element.grid(row=i, column=j, rowspan=rowspan, columnspan=columnspan, **kwargs)
#         self.elements[cad_pos]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
#         return element

#     def get_element(self, cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             return self.elements[cad_pos]["element"]
#         return None
    
#     def is_leaf(self, cad_pos):
#         if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
#             return self.elements[cad_pos]["is_leaf"]
#         return True

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
#     def insert_element_at_end(self, element, is_leaf=False, **kwargs):
#         dim=self.grid_frame.dim
#         if dim[0] == 1 and len(self.elements) == 0:
#             # Significa que no hay ningun elemento todavia (el frame esta inicializado pero no tiene elementos)
#             element.grid(row=0, column=0, columnspan=dim[1], **kwargs)
#             self.elements["0,0"]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
#             return element
#         else:
#             element.grid(row=dim[0], column=0, columnspan=dim[1], **kwargs)
#             self.elements["{},0".format(dim[0])]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
#             aux=np.array([["{},0".format(dim[0])] for i in range(dim[1])], dtype="<U5").T
#             self.grid_frame=GridFrame(dim=(dim[0] + 1, dim[1]), arr=np.concatenate([self.grid_frame.arr, aux], axis=0))
#             return element

#     def number_of_elements(self):
#         return len(list(self.elements.keys()))

#     def get_root(self, elem=None):
#         if elem is None and self.is_root:
#             return self
#         elif elem is not None and elem.is_root:
#             return elem
#         else:
#             if elem is None:
#                 elem=self.father
#             return self.get_root(elem=elem.father)

#     def get_child(self, name):
#         self.search_child(name=name)
#         child=self.child_search
#         self.clear_child_search()
#         return child

#     def search_child(self, name, elem=None):
#         if elem is not None and elem.name == name and self.child_search is None:
#             self.child_search=elem
#         else:
#             if elem is None:
#                 elem=self
#             for cad_pos in elem.elements.keys():
#                 if elem.element_exists(cad_pos=cad_pos):
#                     child=elem.get_element(cad_pos=cad_pos)
#                     is_leaf=elem.is_leaf(cad_pos=cad_pos)
#                     if child is not None and not is_leaf:
#                         self.search_child(name=name, elem=child)

#     def get_child_search(self):
#         return self.child_search
    
#     def clear_child_search(self):
#         self.child_search=None