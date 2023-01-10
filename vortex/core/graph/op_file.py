import os
import cv2
import pydicom
import json


from .op_base import *

class FileOperationReadJSON(UtilOperation, DataSource):

    def __init__(self, path: str, target_keys: str or list = None,  dataset_key: str = "dataset", ):
        super().__init__(OperationTypes.OP_TYPE_NONE, OperationTypes.OP_TYPE_NP_IMAGE)
        f = open(path, "r")
        data = json.load(f)
        self.data = data[dataset_key]
        f.close()

        self.target_keys = target_keys

    def execute(self, index: int, target_keys: str or list = None):
        if target_keys is not None: runtime_target_keys = target_keys
        else: runtime_target_keys = self.target_keys

        sample = self.data[index]

        if isinstance(runtime_target_keys, str):
            x = sample[runtime_target_keys]
        if isinstance(runtime_target_keys, list):
            x = []
            for key in runtime_target_keys:
                x.append(sample[key])
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.execute(index)


class FileOperationLoadImage(UtilOperation):

    def __init__(self, path_image: str = None):
        super().__init__(OperationTypes.OP_TYPE_STRING, OperationTypes.OP_TYPE_NP_IMAGE)
        self.path_image = path_image

    def execute(self, path_image: str = None):
        if path_image is not None: runtime_path_image = path_image
        else: runtime_path_image = self.path_image

        if not os.path.exists(path_image): return None

        if runtime_path_image.endswith(".png") or runtime_path_image.endswith(".jpg"):
            image = cv2.imread(runtime_path_image)
            return image

        elif runtime_path_image.endswith(".dcm"):
            dcm = pydicom.dcmread(runtime_path_image)


