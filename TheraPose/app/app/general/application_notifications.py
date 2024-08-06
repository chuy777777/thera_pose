from general.template_copy import TemplateCopy
from general.application_message_types import MessageTypesConnectionDB

class MessageNotificationConnectionDB(TemplateCopy):
    def __init__(self, message, specific_message, is_exception, message_type):
        TemplateCopy.__init__(self)
        self.message: str=message
        self.specific_message: str=specific_message
        self.is_exception: bool=is_exception
        self.message_type: MessageTypesConnectionDB=message_type

    def __str__(self) -> str:
        return "\nMessage: {}\nSpecific Message: {}\nIs exception: {}\nMessage Type: {}".format(self.message, self.specific_message, self.is_exception, self.message_type)

