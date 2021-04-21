import sys
import atexit

from logipy.importer.file_converter import cleanup
from logipy.logic.rules import add_rule
from logipy.wrappers import LogipyPrimitive as logipy_obj
from logipy.wrappers import logipy_call as logipy_call
import logipy.importer.file_converter
import logipy.importer.gherkin_importer
import logipy.importer.exception_handler


logipy.importer.file_converter.convert_path()
logipy.importer.gherkin_importer.import_gherkin_path()
# TODO: Uncomment cleanup.
# atexit.register(cleanup)

sys.excepthook = logipy.importer.exception_handler.logipy_exception_handler
