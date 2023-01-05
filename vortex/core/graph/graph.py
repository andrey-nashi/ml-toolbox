from .op_file import *

class OperationNode:

    def __init__(self, node_cfg: dict):
        return

class OperationGraph:

    def __init__(self, graph_cfg: list):
        self.dataset_length = 0
        self.graph_cfg = graph_cfg
        self.graph = []

        for node_cfg in graph_cfg:
            node = OperationNode(node_cfg)

        return


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index: int):
        return 0