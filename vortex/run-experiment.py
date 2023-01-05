import json
from core.graph import *


g = OperationGraph()

op_file = FileOperationReadJSON("test.json", "path_image", "target")
g.add_node(op_file, True)

print(len(g))
print(g.__getitem__(0))