import random
from .op_base import *

class UtilOperationRandomize(UtilOperation):
    CODE_SAMPLE_INT = 0
    CODE_SAMPLE_FLOAT = 1
    CODE_RETURN_STORED = 2

    def __init__(self, seed_value: int = None):
        super().__init__(OperationTypes.OP_TYPE_NONE, OperationTypes.OP_TYPE_NUMERIC)
        if seed_value is not None:
            random.seed(seed_value)
        self.storage = None

    def execute(self, command_code=0, min_value: int = 0, max_value: int = 100):
        if command_code == self.CODE_SAMPLE_INT:
            value = random.randint(min_value)