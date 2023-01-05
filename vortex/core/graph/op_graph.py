from .op_factory import *

class OperationGraph:

    def __init__(self, graph_cfg: dict = None):
        self.dataset_length = None
        self.graph_cfg = graph_cfg
        self.graph = []
        self.source_index = None

        if graph_cfg is None:
            return

        for node_cfg in graph_cfg:
            node = OperationNodeFactory.build_from_cfg(node_cfg)
            self.graph.append(node)

    def reset(self):
        self.dataset_length = 0
        self.graph = {}

    def add_node(self, node, is_source_node: bool = False):
        self.graph.append(node)
        if is_source_node:
            self.source_index = len(self.graph) - 1

    def __len__(self):
        if self.dataset_length is not None:
            return self.dataset_length
        else:
            return self.graph[self.source_index].__len__()

    def __getitem__(self, index: int):

        for index in range(0, len(self.graph)):
            node = self.graph[index]
            if index == self.source_index:
                out = node.__getitem__(index)
                print(out)
        return 0