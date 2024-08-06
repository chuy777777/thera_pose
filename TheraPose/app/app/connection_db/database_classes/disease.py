from __future__ import annotations

from bson import ObjectId
from general.template_copy import TemplateCopy

class Disease(TemplateCopy):
    def __init__(self, _id, name):
        TemplateCopy.__init__(self)
        self._id: ObjectId=_id
        self.name: str=name

    def to_json(self):
        return {
            "_id": self._id,
            "name": self.name,
        }
    
    def set_and_get_differences(self, obj: Disease):
        obj_update={
            "_id": self._id,
            "update": {}
        }
        if self.name != obj.name:
            self.name=obj.name
            obj_update["update"]["name"]=obj.name
        return obj_update
    
    # changes -> update (diccionario)
    def apply_changes(self, changes):
        for field in list(changes.keys()):
            if field == "name":
                self.name=changes[field]
        
    @staticmethod
    def from_json(obj):
        return Disease(
            _id=obj["_id"],
            name=obj["name"],
        )

    @staticmethod
    def get_empty():
        obj={
            "_id": ObjectId(),
            "name": "",
        }
        return Disease.from_json(obj=obj)
    
    @staticmethod
    def get_differences(old_obj: Disease, new_obj: Disease):
        obj_update={
            "_id": old_obj._id,
            "update": {}
        }
        if old_obj.name != new_obj.name:
            obj_update["update"]["name"]=new_obj.name
        return obj_update