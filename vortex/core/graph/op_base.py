
class OperationTypes:
    OP_TYPE_NONE = -1
    OP_TYPE_NP_IMAGE = 0
    OP_TYPE_NUMERIC = 1
    OP_TYPE_INT = 2
    OP_TYPE_FLOAT = 3
    OP_TYPE_STRING = 4


class ImageOperation:

    def __init__(self, op_input_type: int, op_output_type: int):
        self.INPUT_TYPE = op_input_type
        self.OUTPUT_TYPE = op_output_type

    def execute(self, **kwargs):
        pass

    @staticmethod
    def apply(**kwargs):
        pass

class UtilOperation:

    def __init__(self, op_input_type: int, op_output_type: int):
        self.INPUT_TYPE = op_input_type
        self.OUTPUT_TYPE = op_output_type

    def execute(self, **kwargs):
        pass


class DataSource:

    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index: int):
        return None