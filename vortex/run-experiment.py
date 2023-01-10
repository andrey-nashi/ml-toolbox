import json
import cv2
from core.graph import *


g = OperationGraph()
print(type(FileOperationLoadImage))
print(type(FileOperationLoadImage()))

op_file = FileOperationReadJSON("test.json", ["path_image", "path_mask"])
g.add_node(op_file, is_source=True)
op_load_input = FileOperationLoadImage()
g.add_node(op_load_input, args_node={"path_image": [0,0]})

print(len(g))
x = g.__getitem__(0)
print(x.shape)
cv2.imwrite("/home/indra/test.png", x)
