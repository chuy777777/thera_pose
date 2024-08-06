from __future__ import annotations

import connection_db.utils_database as utils_database
from bson import ObjectId
from general.template_copy import TemplateCopy

class DatabaseUpdates(TemplateCopy):
    def __init__(self, _id, database_update_date):
        TemplateCopy.__init__(self)
        self._id: ObjectId=_id
        self.database_update_date: str=database_update_date

    def to_json(self):
        return {
            "_id": self._id,
            "database_update_date": self.database_update_date,
        }
    
    def set_database_update_date(self, database_update_date):
        self.database_update_date=database_update_date

    def set_and_get_differences(self, obj: DatabaseUpdates):
        obj_update={
            "_id": self._id,
            "update": {}
        }
        if self.database_update_date != obj.database_update_date:
            self.database_update_date=obj.database_update_date
            obj_update["update"]["database_update_date"]=obj.database_update_date
        return obj_update
    
    # changes -> update (diccionario)
    def apply_changes(self, changes):
        for field in list(changes.keys()):
            if field == "database_update_date":
                self.database_update_date=changes[field]

    @staticmethod
    def from_json(obj):
        return DatabaseUpdates(
            _id=obj["_id"],
            database_update_date=obj["database_update_date"],
        )

    @staticmethod
    def get_empty():
        obj={
            "_id": ObjectId(),
            "database_update_date": utils_database.get_date_today(),
        }
        return DatabaseUpdates.from_json(obj=obj)
    
    @staticmethod
    def get_differences(old_obj: DatabaseUpdates, new_obj: DatabaseUpdates):
        obj_update={
            "_id": old_obj._id,
            "update": {}
        }
        if old_obj.database_update_date != new_obj.database_update_date:
            obj_update["update"]["database_update_date"]=new_obj.database_update_date
        return obj_update