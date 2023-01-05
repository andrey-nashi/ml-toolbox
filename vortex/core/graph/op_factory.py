from .op_image import *
from .op_file import *

OP_IMAGE_CLIP = "op_image_clip"
OP_IMAGE_RESIZE = "op_image_resize"
OP_FILE_READ_JSON = "op_file_read_json"

OP_TABLE = {
    OP_IMAGE_CLIP: ImageOperationClip,
    OP_IMAGE_RESIZE: ImageOperationResize,

    OP_FILE_READ_JSON: FileOperationReadJSON
}

class OperationNodeFactory:

    @staticmethod
    def build_from_cfg(cfg: dict):
        op_name = cfg["op_name"]

        if "op_args" in cfg: op_args = cfg["op_args"]
        else: op_args = {}

        if op_name in OP_TABLE:
            op = OP_TABLE[op_name](**op_args)
            return op
        else:
            raise RuntimeError('Unknown operation ', op_name)
