from pymongo import MongoClient, errors, timeout
from pymongo.collection import Collection
import copy

import connection_db.utils_database as utils_database
from general.application_notifications import MessageNotificationConnectionDB
from general.application_message_types import MessageTypesConnectionDB
from connection_db.db import DB

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 

class ConnectionDB():
    def __init__(self, database_name, mongodb_uri, connection_type: Literal["local", "cloud"], **kwargs):
        self.database_name=database_name
        self.mongodb_uri=mongodb_uri
        self.connection_type=connection_type
        self.other_conection_type='local' if connection_type == 'cloud' else 'cloud'
        self.operation_timeout_s=5
        self.connection_timeout_s=5
        self.message_notification_connection_db: MessageNotificationConnectionDB=None
        self.init_variables()
        self.get_username_password()

    def init_variables(self):
        self.db=DB()
        self.other_db=DB()
        self.db_collection_names=list(self.db.collections.keys())
        self.mongodb_collection_names=[]
        self.mongodb_collections={}
        self.client=None
        self.session=None
        self.client_connected=False

    def get_username_password(self):
        username,password=self.mongodb_uri.split("//")[1].split("@")[0].split(":")
        self.username=username
        self.password=password

    """
    Es posible que la base de datos deje de ser inadecuado despues de crear la conexion.
        - Es inadecuada si no contiene las colecciones que debe contener o si esta vacia.
        - Es adecuada si contiene las colecciones que debe tener.
    """
    def is_inadequate_database(self):
        if len(self.db_collection_names) == len(self.mongodb_collection_names) and len(list(set(self.db_collection_names).difference(set(self.mongodb_collection_names)))) == 0:
            return False
        else:
            return True
    
    def operation(self, op: Literal["create_connection", "save", "build_database_classes", "load_database", "close"], **kwargs):
        try:
            if op == "create_connection":

                if not self.client_connected:
                    if self.mongodb_uri != "": 
                        # Creamos la clase de conexion
                        self.client=MongoClient(host=self.mongodb_uri, connectTimeoutMS=self.connection_timeout_s*1000, serverSelectionTimeoutMS=self.connection_timeout_s*1000)
                        # Para tener referencia a la base de datos
                        mongodb=self.client[self.database_name]
                        # Si existe conexion debe poder ejecutarse la siguiente linea 
                        self.mongodb_collection_names=mongodb.list_collection_names()
                        # Si existe conexion, entonces proseguimos con las siguientes lineas de codigo
                        self.client_connected=True
                        # Obtenemos las colecciones de la base de datos (las cuales tiene conexion con la base de datos)
                        for collection_name in self.db_collection_names:
                            self.mongodb_collections[collection_name]=mongodb[collection_name]
                         # Cargamos el diccionario que guarda los cambios no guardados en la base de datos
                        self.db.load_db_notifications(connection_type=self.connection_type)
                        self.other_db.load_db_notifications(connection_type=self.other_conection_type)
                        self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Todo ha salido bien. La conexion se ha creada exitosamente para la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.CREATE_CONNECTION_OK)
                    else:
                        self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="No se ha proporcionado una URI para conectar con la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.NOT_URI)
                else:
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Ya existe una conexion para la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.ALREADY_CLIENT_CONNECTED)

            elif op == "save":
                
                # A la hora de guardar solo es requerido el diccionario de cambios pendientes (no las clases)
                if self.client_connected:
                    # Verificar si hay algo que guardar
                    if not self.db.is_empty_db_notifications():
                        with timeout(self.operation_timeout_s):
                            self.session=self.client.start_session(causal_consistency=True)
                            self.session.start_transaction()
                            # Actualizamos la fecha de actualizacion de la base de datos en la base de datos
                            database_update_date=utils_database.get_date_today()
                            self.mongodb_collections["DatabaseUpdates"].update_one({"_id": self.db.collections["DatabaseUpdates"]._id}, {"$set": {"database_update_date": database_update_date}}, session=self.session)
                            # Realizamos los respectivos cambios en la base de datos (inserciones, eliminaciones y modificaciones)
                            for collection_name in list(self.db.db_notifications.keys()):
                                collection_notifications=self.db.db_notifications[collection_name]
                                collection_mongo: Collection=self.mongodb_collections[collection_name]
                                if len(collection_notifications["insert"]) > 0:
                                    documents=list(map(lambda elem: elem.to_json(), collection_notifications["insert"]))
                                    collection_mongo.insert_many(documents, session=self.session)
                                if len(collection_notifications["delete"]) > 0:
                                    _ids=collection_notifications["delete"]
                                    collection_mongo.delete_many({"_id": {"$in": _ids}}, session=self.session)
                                if len(collection_notifications["update"]) > 0:
                                    for elem in collection_notifications["update"]:
                                        collection_mongo.update_one({"_id": elem["_id"]}, {"$set": elem["update"]}, session=self.session)   
                            self.session.commit_transaction()
                            # Actualizamos la fecha de actualizacion de la base de datos en la clase
                            self.db.collections["DatabaseUpdates"].set_database_update_date(database_update_date=database_update_date)
                            # Inicializamos el diccionario de cambios y eliminamos su archivo (si es que existe)
                            self.db.init_db_notifications()
                            self.db.delete_file_db_notifications(connection_type=self.connection_type)
                            self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Todo ha salido bien. Se ha guardado exitosamente en la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.SAVE_OK)
                    else:
                        self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="No hay nada que guardar por el momento en la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.ALREADY_SAVED)
                else:
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="No fue posible realizar la operacion de guardar en la base de datos '{}'. No existe una conexion.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.NOT_CLIENT_CONNECTED)

            elif op == "build_database_classes":
                
                if self.client_connected:
                    if not self.is_inadequate_database():
                        if not self.db.is_built:
                            with timeout(self.operation_timeout_s):
                                mongodb=self.client[self.database_name]
                                for collection_name in self.db_collection_names:
                                    collection_mongo=mongodb[collection_name]
                                    if collection_name in ["DatabaseUpdates"]:
                                        self.db.create_collection(collection_name=collection_name, result=collection_mongo.find_one({},{}))
                                    else:
                                        self.db.create_collection(collection_name=collection_name, result=collection_mongo.find({},{}))
                                self.db.is_built=True
                                # Cargar las cambios pendientes (que se encuentran en el diccionario, si es que no esta vacio) en las clases construidas
                                if not self.db.is_empty_db_notifications():
                                    self.db.load_db_notifications_to_classes()
                                self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Todo ha salido bien. Las clases de la base de datos '{}' se han construido exitosamente.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.BUILD_DATABASE_CLASSES_OK)
                        else:
                            # No es necesario construir las clases porque ya se realizo esta tarea
                            self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Las clases de la base de datos '{}' ya estan construidas. No fue necesario volver a construirlas.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.DATABASES_ARE_ALREADY_BUILT)
                    else:
                        self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="No fue posible construir las clases de la base de datos '{}'. La base de datos no esta construida de manera adecuada.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.INADEQUATE_DATABASE)
                else:
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="No fue posible construir las clases de la base de datos '{}'. No existe una conexion.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.NOT_CLIENT_CONNECTED)

            elif op == "load_database":
                
                if self.client_connected:
                    if not self.is_inadequate_database():
                        # Crearemos la conexion con la otra base de datos 
                        # Variables de base de datos
                        username=kwargs["username"]
                        password=kwargs["password"]
                        connection_db=ConnectionDB.create_connection_db(username=username, password=password, connection_type=self.other_conection_type)
                        connection_db.operation(op="create_connection")
                        if connection_db.message_notification_connection_db.message_type == MessageTypesConnectionDB.CREATE_CONNECTION_OK:
                            if len(connection_db.mongodb_collection_names) == 0 or not connection_db.is_inadequate_database():
                                with timeout(connection_db.operation_timeout_s):
                                    connection_db.session=connection_db.client.start_session(causal_consistency=True)
                                    connection_db.session.start_transaction()

                                    mongodb=connection_db.client[self.database_name]
                                    is_created_db=False if len(connection_db.mongodb_collection_names) == 0 else True

                                    for collection_name in self.db_collection_names:
                                        mongodb_collection=None
                                        if is_created_db:
                                            # Limpiamos primero la coleccion de la base de datos destino
                                            mongodb_collection: Collection=connection_db.mongodb_collections[collection_name]
                                            mongodb_collection.delete_many({}, session=connection_db.session)
                                        else:
                                            mongodb_collection=mongodb.create_collection(name=collection_name, session=connection_db.session)

                                        if collection_name in ["DatabaseUpdates"]:
                                            obj=self.db.collections[collection_name]
                                            mongodb_collection.insert_one(obj.to_json(), session=connection_db.session)
                                        else:
                                            l=self.db.collections[collection_name]
                                            list_documents=list(map(lambda elem: elem.to_json(), l))
                                            mongodb_collection.insert_many(list_documents, session=connection_db.session)

                                    connection_db.session.commit_transaction()
                                    # Cerramos la sesion de la base de datos destino (ya que con esta no estamos trabajando)
                                    connection_db.operation(op="close")

                                    # Inicializamos el diccionario de cambios de la base de datos destino
                                    self.other_db.init_db_notifications()
                                    # Eliminamos el diccionario de cambios de la base de datos destino (ya que tenemos todo actualizado)
                                    self.other_db.delete_file_db_notifications(connection_type=self.other_conection_type)
                                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="Todo ha salido bien. Se ha cargado exitosamente la base de datos '{}' a la base de datos '{}'.".format(self.connection_type, connection_db.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.LOAD_DATABASE_OK)
                            else:
                                self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="No fue posible realizar la operacion de cargar la base de datos '{}' a la base de datos '{}'. La base de datos '{}' no esta construida de manera adecuada.".format(self.connection_type, connection_db.connection_type, connection_db.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.INADEQUATE_DATABASE)
                        else:
                            self.message_notification_connection_db=connection_db.message_notification_connection_db
                    else:
                        self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente conectado.", specific_message="No fue posible realizar la operacion de cargar la base de datos '{}' a la base de datos '{}'. La base de datos '{}' no esta construida de manera adecuada.".format(self.connection_type, connection_db.connection_type, self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.INADEQUATE_DATABASE)
                else:
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="No fue posible realizar la operacion de cargar la base de datos '{}' a la base de datos '{}'. El cliente no esta conectado.".format(self.connection_type, connection_db.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.NOT_CLIENT_CONNECTED)

            elif op == "close":

                if self.client_connected:
                    # Cerramos la conexion
                    self.client.close()
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="Todo ha salido bien. La conexion se ha cerrado exitosamente para la base de datos '{}'.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.CLOSE_OK)
                else:
                    self.message_notification_connection_db=MessageNotificationConnectionDB(message="Cliente desconectado.", specific_message="No fue posible realizar la operacion de cerrar la conexion para la base de datos '{}'. No existe una conexion.".format(self.connection_type), is_exception=False, message_type=MessageTypesConnectionDB.NOT_CLIENT_CONNECTED)
                # Es importante guardar este diccionario si es que tiene cambios aun pendientes
                if not self.db.is_empty_db_notifications():
                    self.db.save_db_notifications(connection_type=self.connection_type)
                if not self.other_db.is_empty_db_notifications():
                    self.other_db.save_db_notifications(connection_type=self.other_conection_type)
                self.init_variables()

        except errors.ServerSelectionTimeoutError as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema. No se ha podido conectar con la base de datos.\nServerSelectionTimeoutError: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.SERVER_SELECTION_TIMEOUT_ERROR)
        except errors.OperationFailure as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema. Ha fallado alguna operacion.\nOperationFailure: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.OPERATION_FAILURE)
        except errors.InvalidOperation as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema. Ha habido una operacion invalida.\nInvalidOperation: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.INVALID_OPERATION)
        except errors.AutoReconnect as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema. Se ha perdido la conexion con la base de datos y la conexion automatica no fue exitosa.\nAutoReconnect: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.AUTO_RECONNECT)
        except errors.ConnectionFailure as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema. No se pudo establecer una conexión a la base de datos o se ha perdido la conexion.\nConnectionFailure: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.CONNECTION_FAILURE)
        except errors.PyMongoError as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema.\nPyMongoError: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.PY_MONGO_ERROR)
        except Exception as e:
            self.message_notification_connection_db=MessageNotificationConnectionDB(message="" if self.message_notification_connection_db is None else self.message_notification_connection_db.message, specific_message="Ha ocurrido un problema.\nException: {}".format(e), is_exception=True, message_type=MessageTypesConnectionDB.EXCEPTION)
        finally:
            if self.session is not None:
                self.session.end_session()
                self.session=None

    def collection_notification_insert(self, collection_name, obj):
        self.db.collection_notification_insert(collection_name=collection_name, obj=obj)
        self.other_db.collection_notification_insert(collection_name=collection_name, obj=obj)

    def collection_notification_delete(self, collection_name, _id):
        self.db.collection_notification_delete(collection_name=collection_name, _id=_id)
        self.other_db.collection_notification_delete(collection_name=collection_name, _id=_id)
    
    def collection_notification_update(self, collection_name, obj_update):
        updated_registry=self.db.collection_notification_update(collection_name=collection_name, obj_update=obj_update)
        self.other_db.collection_notification_update(collection_name=collection_name, obj_update=obj_update)
        return updated_registry

    @staticmethod
    def create_connection_db(username, password, connection_type):
        database_name="therapose"
        replica_set_name="dbrs"
        # URIs para las conexiones con las bases de datos
        mongodb_uri=""
        if connection_type == "local":
            # URI para crear conexion con 'local'
            # mongodb_uri="mongodb://{}:{}@localhost:30001/replicaSet={}?authSource=admin&replicaSet={}&readPreference=primary&directConnection=true&ssl=false".format(username, password, replica_set_name, replica_set_name) 
            mongodb_uri="mongodb://{}:{}@mongodb1:27017/replicaSet={}?authSource=admin&replicaSet={}&readPreference=primary&directConnection=true&ssl=false".format(username, password, replica_set_name, replica_set_name) 
        elif connection_type == "cloud":
            # URI para crear conexion con 'cloud'
            mongodb_uri="mongodb+srv://{}:{}@cluster0.zjmv5mw.mongodb.net/{}?retryWrites=true&w=majority".format(username, password, database_name)
        connection_db=ConnectionDB(database_name=database_name, mongodb_uri=mongodb_uri, connection_type=connection_type)
        return connection_db
                        
