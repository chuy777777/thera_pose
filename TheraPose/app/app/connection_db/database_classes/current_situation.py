from __future__ import annotations

from bson import ObjectId
from general.template_copy import TemplateCopy

class CurrentSituation(TemplateCopy):
    def __init__(self, _id, name, color):
        TemplateCopy.__init__(self)
        self._id: ObjectId=_id
        self.name: str=name
        self.color: list=color

    def to_json(self):
        return {
            "_id": self._id,
            "name": self.name,
            "color": self.color,
        }
    
    def set_and_get_differences(self, obj: CurrentSituation):
        obj_update={
            "_id": self._id,
            "update": {}
        }
        if self.name != obj.name:
            self.name=obj.name
            obj_update["update"]["name"]=obj.name
        if self.color != obj.color:
            self.color=obj.color
            obj_update["update"]["color"]=obj.color
        return obj_update

    # changes -> update (diccionario)
    def apply_changes(self, changes):
        for field in list(changes.keys()):
            if field == "name":
                self.name=changes[field]
            elif field == "color":
                self.color=changes[field]

    @staticmethod
    def from_json(obj):
        return CurrentSituation(
            _id=obj["_id"],
            name=obj["name"],
            color=obj["color"],
        )

    @staticmethod
    def get_empty():
        obj={
            "_id": ObjectId(),
            "name": "",
            "color": [0,0,0],
        }
        return CurrentSituation.from_json(obj=obj)
    
    @staticmethod
    def get_differences(old_obj: CurrentSituation, new_obj: CurrentSituation):
        obj_update={
            "_id": old_obj._id,
            "update": {}
        }
        if old_obj.name != new_obj.name:
            obj_update["update"]["name"]=new_obj.name
        if old_obj.color != new_obj.color:
            obj_update["update"]["color"]=new_obj.color
        return obj_update