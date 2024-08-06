try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 
    
class BuildDBComponent():
    def __init__(self):
        pass

    # Este metodo se debe sobreescribir 
    def build_db_component(self, collection_name, registry, op: Literal["insert", "update", "delete"], **kwargs):
        if op == "insert":
            pass
        elif op == "update":
            pass
        elif op == "delete":
            pass

    # Este metodo se debe sobreescribir 
    def listener_db_component(self, collection_name, registry, **kwargs):
        pass