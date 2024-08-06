from __future__ import annotations

import connection_db.utils_database as utils_database
from bson import ObjectId
from general.template_copy import TemplateCopy
    
class Patient(TemplateCopy):
    def __init__(self, _id, name, session, pathologies, schedule, diseases, current_situation):
        TemplateCopy.__init__(self)
        self._id: ObjectId=_id
        self.name: str=name
        self.session: ObjectId=session
        self.pathologies: list=pathologies
        self.schedule: list=schedule
        self.diseases: list=diseases
        self.current_situation: ObjectId=current_situation

    def to_json(self):
        return {
            "_id": self._id,
            "name": self.name,
            "session": self.session,
            "pathologies": self.pathologies,
            "schedule": self.schedule,
            "diseases": self.diseases,
            "current_situation": self.current_situation,
        }

    def set_and_get_differences(self, obj: Patient):
        obj_update={
            "_id": self._id,
            "update": {}
        }
        if self.name != obj.name:
            self.name=obj.name
            obj_update["update"]["name"]=obj.name
        if self.session != obj.session:
            self.session=obj.session
            obj_update["update"]["session"]=obj.session
        if len(self.pathologies) != len(obj.pathologies) or self.pathologies != obj.pathologies:
            self.pathologies=obj.pathologies
            obj_update["update"]["pathologies"]=obj.pathologies
        if len(self.schedule) != len(obj.schedule) or self.schedule != obj.schedule:
            self.schedule=obj.schedule
            obj_update["update"]["schedule"]=obj.schedule 
        if len(self.diseases) != len(obj.diseases) or self.diseases != obj.diseases:
            self.diseases=obj.diseases
            obj_update["update"]["diseases"]=obj.diseases
        if self.current_situation != obj.current_situation:
            self.current_situation=obj.current_situation
            obj_update["update"]["current_situation"]=obj.current_situation
        return obj_update

    # changes -> update (diccionario)
    def apply_changes(self, changes):
        for field in list(changes.keys()):
            if field == "name":
                self.name=changes[field]
            elif field == "session":
                self.session=changes[field]
            elif field == "pathologies":
                self.pathologies=changes[field]
            elif field == "schedule":
                self.schedule=changes[field]
            elif field == "diseases":
                self.diseases=changes[field]
            elif field == "current_situation":
                self.current_situation=changes[field]

    @staticmethod
    def from_json(obj):
        return Patient(
            _id=obj["_id"],
            name=obj["name"],
            session=obj["session"],
            pathologies=obj["pathologies"],
            schedule=obj["schedule"],
            diseases=obj["diseases"],
            current_situation=obj["current_situation"]
        )

    @staticmethod
    def get_empty():
        obj={
            "_id": ObjectId(),
            "name": "",
            "session": None,
            "pathologies": [],
            "schedule": [],
            "diseases": [],
            "current_situation": None
        }
        return Patient.from_json(obj=obj)
    
    @staticmethod
    def get_differences(old_obj: Patient, new_obj: Patient):
        obj_update={
            "_id": old_obj._id,
            "update": {}
        }
        if old_obj.name != new_obj.name:
            obj_update["update"]["name"]=new_obj.name
        if old_obj.session != new_obj.session:
            obj_update["update"]["session"]=new_obj.session
        if len(old_obj.pathologies) != len(new_obj.pathologies) or old_obj.pathologies != new_obj.pathologies:
            obj_update["update"]["pathologies"]=new_obj.pathologies
        if len(old_obj.schedule) != len(new_obj.schedule) or old_obj.schedule != new_obj.schedule:
            obj_update["update"]["schedule"]=new_obj.schedule 
        if len(old_obj.diseases) != len(new_obj.diseases) or old_obj.diseases != new_obj.diseases:
            obj_update["update"]["diseases"]=new_obj.diseases
        if old_obj.current_situation != new_obj.current_situation:
            obj_update["update"]["current_situation"]=new_obj.current_situation
        return obj_update
    
