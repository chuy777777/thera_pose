import pickle
import os
from bson.objectid import ObjectId

from connection_db.database_classes.database_updates import DatabaseUpdates
from connection_db.database_classes.session import Session
from connection_db.database_classes.current_situation import CurrentSituation
from connection_db.database_classes.patient import Patient
from connection_db.database_classes.disease import Disease
from connection_db.database_classes.pathology import Pathology

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# Directorio desde donde se ejecuta la aplicacion
global_path_app=os.path.dirname(os.path.abspath('__file__'))
# Directorio donde se guarda todo lo relacionado a archivos temporales con respecto a la base de datos
global_path_temp=os.path.join(global_path_app, "connection_db/temp")

class DB():
    def __init__(self):
        self.init_collections()
        self.init_db_notifications()
        self.is_built=False

    def init_collections(self):
        self.collections={
            "DatabaseUpdates": None,
            "Session": [],
            "CurrentSituation": [], 
            "Patient": [],
            "Disease": [],
            "Pathology": [], 
        }

    # Aqui se registra toda operacion realizada y aun no guardada en la base de datos local/nube
    def init_db_notifications(self):
        self.db_notifications={
            "DatabaseUpdates": {"insert": [], "update": [], "delete": []},
            "Session": {"insert": [], "update": [], "delete": []},
            "CurrentSituation": {"insert": [], "update": [], "delete": []},
            "Patient": {"insert": [], "update": [], "delete": []},
            "Disease": {"insert": [], "update": [], "delete": []},
            "Pathology": {"insert": [], "update": [], "delete": []},
        }

    def create_collection(self, collection_name, result):
        if collection_name=="DatabaseUpdates":
            self.collections[collection_name]=DatabaseUpdates.from_json(obj=result)
        elif collection_name=="Session":
            self.collections[collection_name]=list(map(lambda elem: Session.from_json(obj=elem), result))
        elif collection_name=="CurrentSituation":
            self.collections[collection_name]=list(map(lambda elem: CurrentSituation.from_json(obj=elem), result))
        elif collection_name=="Patient":
            self.collections[collection_name]=list(map(lambda elem: Patient.from_json(obj=elem), result))
        elif collection_name=="Disease":
            self.collections[collection_name]=list(map(lambda elem: Disease.from_json(obj=elem), result))
        elif collection_name=="Pathology":
            self.collections[collection_name]=list(map(lambda elem: Pathology.from_json(obj=elem), result))
       
    def is_empty_db_notifications(self):
        for collection_name in list(self.db_notifications.keys()):
            collection_notifications=self.db_notifications[collection_name]
            if len(collection_notifications["insert"]) > 0 or len(collection_notifications["delete"]) > 0 or len(collection_notifications["update"]) > 0:
                return False
        return True

    """
    Se carga el diccionario (si es que existe) que registra los cambios que se han realizado y aun
    no se han guardado en la base de datos (esto debe hacerse siempre al inicio).

    Esta operacion no require de conexion, ya que solo carga un diccionario que se encuentra 
    almacenado enla computadora (si no existe, se crea uno vacio).

    Si existe el archivo:
        - Significa que hay informacion pendiente por guardar en la base de datos.
        - Se necesita cargar dichos cambios a las clases construidas (ya que tendran informacion antigua).
    Si no existe el archivo:  
        - Significa que no hay nada que guardar en la base de datos (todo esta actualizado).  
        - Se continua normal (no se hace nada).
    """
    def load_db_notifications(self, connection_type):
        path="{}/{}/db_notifications.dat".format(global_path_temp, connection_type)
        try:
            with open(path, "rb") as f:
                self.db_notifications=pickle.load(f)
        except (OSError, IOError) as e:
            self.init_db_notifications()

    def save_db_notifications(self, connection_type):
        dirname="{}/{}".format(global_path_temp, connection_type)
        path="{}/db_notifications.dat".format(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as f:
            pickle.dump(self.db_notifications, f)

    def delete_file_db_notifications(self, connection_type):
        dirname="{}/{}".format(global_path_temp, connection_type)
        path="{}/db_notifications.dat".format(dirname)
        if os.path.exists(path):
            os.remove(path)

    def load_db_notifications_to_classes(self):
        for collection_name in list(self.db_notifications.keys()):
            collection_notifications=self.db_notifications[collection_name]
            if len(collection_notifications["insert"]) > 0:
                for obj in collection_notifications["insert"]:
                    self.collections[collection_name].append(obj)
            if len(collection_notifications["delete"]) > 0:
                _ids=collection_notifications["delete"]
                self.collections[collection_name]=list(filter(lambda elem: elem._id not in _ids, self.collections[collection_name]))
            if len(collection_notifications["update"]) > 0:
                for elem in collection_notifications["update"]:
                    obj=self.find_by_id(collection_name=collection_name, _id=elem["_id"])
                    if obj is not None:
                        obj.apply_changes(changes=elem["update"])

    def find_by_id(self, collection_name: Literal["Session", "CurrentSituation", "Patient", "Disease", "Pathology", "Game", "Algorithm"], _id: ObjectId):
        if self.is_built:
            list_registry=self.collections[collection_name]
            for i in range(len(list_registry)):
                registry=list_registry[i]
                if registry._id == _id:
                    return registry
        return None
    
    """
    NOTA: 
        Los metodos ya hacen las respectivas modificaciones:
            - Si ya estan construidas las clases:
                - collections
                - db_notifications
            - Si no estan construidas las clases:
                - db_notifications

            - collection_notification_insert
            - collection_notification_delete
            - collection_notification_update
    """

    """
    Aqui simplemente insertamos (no hay mas).
    """
    def collection_notification_insert(self, collection_name, obj):
        self.db_notifications[collection_name]["insert"].append(obj)
        if self.is_built:
            self.collections[collection_name].append(obj)

    """
    Hacemos lo siguiente:
        - Si el registro a eliminar se encuentra en las notificaciones de insercion (es decir, aun 
          no se encuentra guardado en la base de datos), se elimina dicha notificacion de insercion
          y se sale del metodo.
        - Si el registro a eliminar se encuentra en las notificaciones de modificacion (es decir, aun 
          no se encuentra guardado en la base de datos), se elimina dicha notificacion de modificacion
          y se agrega dicho registro a las notificaciones de eliminacion.
        - Si el registro a eliminar no tiene notificaciones de insercion ni de modificacion, solo 
          se agrega dicho registro a las notificaciones de eliminacion.
    """
    def collection_notification_delete(self, collection_name, _id):
        elem_temp=None
        list_temp=[]
        for elem in self.db_notifications[collection_name]["insert"]:
            if elem._id == _id:
                elem_temp=elem
            else:
                list_temp.append(elem)
        if elem_temp is not None:
            self.db_notifications[collection_name]["insert"]=list_temp
            if self.is_built:
                self.collections[collection_name]=list(filter(lambda elem: elem._id != _id, self.collections[collection_name]))
            return
        
        list_temp=[]
        for elem in self.db_notifications[collection_name]["update"]:
            if elem["_id"] == _id:
                elem_temp=elem
            else:
                list_temp.append(elem)
        if elem_temp is not None:
            self.db_notifications[collection_name]["update"]=list_temp

        self.db_notifications[collection_name]["delete"].append(_id) 
        if self.is_built:
            self.collections[collection_name]=list(filter(lambda elem: elem._id != _id, self.collections[collection_name]))
            
    """
    'obj_update' es de la siguiente forma:
        obj_update={
            "_id": <_id>,
            "update": {}
        }

    El 'update' contiene <field>: <value>
        - <filed>: El nombre del atributo que se modifico
            Para atributos compuestos (ejemplo):
                filed_1.field_1_1
        - <value>: El nuevo valor del atributo

    Hacemos lo siguiente:
        - Si el registro a modificar se encuentra en las notificaciones de insercion (es decir, aun 
          no se encuentra guardado en la base de datos), se modifica dicha notificacion de insercion
          y se sale del metodo.
        - Si el registro a modificar se encuentra en las notificaciones de modificacion (es decir, aun 
          no se encuentra guardado en la base de datos), se modifica dicha notificacion de modificacion
          con:
            - Valores nuevos a claves ya existentes
            - Claves nuevas y sus respectivos valores
          y se sale del metodo.
        - Si el registro a modificar no tiene notificaciones de insercion ni de modificacion, solo 
          se agrega dicho registro a las notificaciones de modificacion.
    """
    def collection_notification_update(self, collection_name, obj_update):
        if len(obj_update["update"]) > 0:
            obj=None
            if self.is_built:
                obj=self.find_by_id(collection_name=collection_name, _id=obj_update["_id"])
                obj.apply_changes(changes=obj_update["update"])

            l=self.db_notifications[collection_name]["insert"]
            for i in range(len(l)):
                if l[i]._id == obj_update["_id"]:
                    # Actualizar el 'insert'
                    l[i].apply_changes(changes=obj_update["update"])
                    return l[i].copy()

            l=self.db_notifications[collection_name]["update"]
            for i in range(len(l)):
                if l[i]["_id"] == obj_update["_id"]:
                    # Esto significa que un objeto modificado ha sido modificado de nuevo
                    s1=set(l[i]["update"].keys())
                    s2=set(obj_update["update"].keys())
                    # z = x.difference(y)
                    # Devuelve un conjunto que contiene los elementos que solo existen en el conjunto 'x' y no en el conjunto 'y'
                    # z = x.intersection(y)
                    # Devuelve un conjunto que contiene los elementos que existen tanto en el conjunto x como en el conjunto y
                    new_set=s2.difference(s1)
                    old_set=s1.intersection(s2)
                    for field in list(new_set):
                        l[i]["update"][field]=obj_update["update"][field]
                    for field in list(old_set):
                        l[i]["update"][field]=obj_update["update"][field]
                    return None if obj is None else obj.copy()
                
            self.db_notifications[collection_name]["update"].append(obj_update)
            return None if obj is None else obj.copy()
        return None




           