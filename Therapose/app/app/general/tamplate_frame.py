import numpy as np
import copy

from general.grid_frame import GridFrame

class TemplateFrame():
    def __init__(self, father, grid_frame, is_root, name, grid_information):
        self.father=father
        self.grid_frame=grid_frame
        self.is_root=is_root
        self.name=name
        self.grid_information=grid_information
        self.child_search=None
        self.elements={}
        self.is_visible=True

    # Este metodo se debe sobreescribir
    def create_specific_grid_frame(self, grid_frame: GridFrame):
        pass

    # Este metodo se debe sobreescribir
    def hide_frame(self):
        pass

    # Este metodo se debe sobreescribir
    def show_frame(self):
        pass

    # Este metodo se debe sobreescribir
    def toggle_visibility(self):
        pass
    
    # Este metodo nos permite verificar si ha sido destruido el elemento o no
    def element_exists(self, cad_pos):
        return self.key_exists(cad_pos=cad_pos) and self.elements[cad_pos]["element"].winfo_exists()
    
    # Este metodo nos permite verificar si una clave existe en el diccionario
    def key_exists(self, cad_pos):
        return cad_pos in list(self.elements.keys())
    
    # Este metodo se debe sobreescribir
    def enable_fixed_size(self):
        pass

    # Este metodo se debe sobreescribir
    def desable_fixed_size(self):
        pass

    def insert_element(self, cad_pos, element, **kwargs):
        if element is not None:
            is_leaf=False if hasattr(element, 'grid_frame') else True
            i,j=[int(val) for val in cad_pos.split(",")]
            columnspan=self.grid_frame.dict[cad_pos]["columnspan"]
            rowspan=self.grid_frame.dict[cad_pos]["rowspan"]
            element.grid(row=i, column=j, rowspan=rowspan, columnspan=columnspan, **kwargs)
            self.elements[cad_pos]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
            return element
        return None

    def get_element(self, cad_pos):
        if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
            return self.elements[cad_pos]["element"]
        return None
    
    def is_leaf(self, cad_pos):
        if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
            return self.elements[cad_pos]["is_leaf"]
        return True

    def hide_element(self, cad_pos):
        if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
            self.elements[cad_pos]["element"].grid_forget()
            if hasattr(self.elements[cad_pos]["element"], 'is_visible'):
                self.elements[cad_pos]["element"].is_visible=False

    def show_element(self, cad_pos):
        if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
            i,j=[int(val) for val in cad_pos.split(",")]
            columnspan=self.grid_frame.dict[cad_pos]["columnspan"]
            rowspan=self.grid_frame.dict[cad_pos]["rowspan"]
            self.elements[cad_pos]["element"].grid(row=i, column=j, rowspan=rowspan, columnspan=columnspan, **self.elements[cad_pos]["kwargs"])
            if hasattr(self.elements[cad_pos]["element"], 'is_visible'):
                self.elements[cad_pos]["element"].is_visible=True

    def destroy_element(self, cad_pos):
        if self.key_exists(cad_pos=cad_pos) and self.element_exists(cad_pos=cad_pos):
            self.elements[cad_pos]["element"].destroy()
            del self.elements[cad_pos]

    def destroy_all(self):
        keys=list(self.elements.keys())
        for cad_pos in keys:
            self.destroy_element(cad_pos=cad_pos)

    # Solo utilizar cuando se trabaje con listas
    def insert_element_at_end(self, element, **kwargs):
        if element is not None:
            is_leaf=True if hasattr(element, 'grid_frame') else False
            dim=self.grid_frame.dim
            if dim[0] == 1 and len(self.elements) == 0:
                # Significa que no hay ningun elemento todavia (el frame esta inicializado pero no tiene elementos)
                element.grid(row=0, column=0, columnspan=dim[1], **kwargs)
                self.elements["0,0"]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
                return element
            else:
                element.grid(row=dim[0], column=0, columnspan=dim[1], **kwargs)
                self.elements["{},0".format(dim[0])]={"element": element, "is_leaf": is_leaf, "kwargs": kwargs}
                aux=np.array([["{},0".format(dim[0])] for i in range(dim[1])], dtype="<U5").T
                self.grid_frame=GridFrame(dim=(dim[0] + 1, dim[1]), arr=np.concatenate([self.grid_frame.arr, aux], axis=0))
                return element
        return None

    def number_of_elements(self):
        return len(list(self.elements.keys()))

    def get_root(self, elem=None):
        if elem is None:
            elem=self
        if elem.is_root:
            return elem
        else:
            return self.get_root(elem=elem.father)

    # Este metodo es el que se debe usar
    def get_child(self, name):
        self.search_child(name=name)
        child=self.child_search
        self.clear_child_search()
        return child

    def search_child(self, name, elem=None):
        if self.name == name:
            self.child_search=self
        elif elem is not None and elem.name == name and self.child_search is None:
            self.child_search=elem
        else:
            if elem is None:
                elem=self
            for cad_pos in elem.elements.keys():
                if elem.element_exists(cad_pos=cad_pos):
                    child=elem.get_element(cad_pos=cad_pos)
                    is_leaf=elem.is_leaf(cad_pos=cad_pos)
                    if child is not None and not is_leaf:
                        self.search_child(name=name, elem=child)

    def get_child_search(self):
        return self.child_search
    
    def clear_child_search(self):
        self.child_search=None
