import json
from .op_base import *

class FileOperationReadJSON(UtilOperation, DataSource):

    def __init__(self, path: str, target_keys_in: str or list = None, target_keys_out: str or list = None,  dataset_key: str = "dataset", ):
        f = open(path, "r")
        data = json.load(f)
        print(data["dataset"])
        print(dataset_key)
        self.data = data[dataset_key]
        f.close()



        self.target_keys_in = target_keys_in
        self.target_keys_out = target_keys_out

    def execute(self, index: int, target_keys_in: str or list = None, target_keys_out: str or list = None):
        if target_keys_in is not None: runtime_target_keys_in = target_keys_in
        else: runtime_target_keys_in = self.target_keys_in

        if target_keys_out is not None: runtime_target_keys_out = target_keys_out
        else: runtime_target_keys_out = self.target_keys_out

        sample = self.data[index]

        x_in = None
        x_out = None

        if isinstance(runtime_target_keys_in, str):
            x_in = sample[runtime_target_keys_in]
        if isinstance(runtime_target_keys_in, list):
            x_in = []
            for key in runtime_target_keys_in:
                x_in.append(sample[key])

        if isinstance(runtime_target_keys_out, str):
            x_out = sample[runtime_target_keys_out]
        if isinstance(runtime_target_keys_out, list):
            x_out = []
            for key in runtime_target_keys_out:
                x_out.append(sample[key])

        return x_in, x_out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.execute(index)





