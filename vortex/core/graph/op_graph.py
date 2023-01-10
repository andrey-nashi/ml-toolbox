from .op_factory import *



class OperationGraph:

    def __init__(self, graph_cfg: dict = None):
        self.dataset_length = None
        self.graph_cfg = graph_cfg
        self.graph_nodes = []
        self.graph_connectors = []
        self.source_index = None

        if graph_cfg is None:
            return


    def reset(self):
        self.dataset_length = 0
        self.graph_nodes = {}

    def add_node(self, node, args_node: dict = {}, args_static: dict = {}, is_source: bool = False):
        self.graph_nodes.append(node)
        if is_source:
            self.source_index = len(self.graph_nodes) - 1
            self.graph_connectors.append(None)
        else:
            self.graph_connectors.append([args_node, args_static])

    def __len__(self):
        if self.dataset_length is not None:
            return self.dataset_length
        else:
            return self.graph_nodes[self.source_index].__len__()

    def __getitem__(self, sample_index: int):

        data_stack = {}
        data_stack[self.source_index] = self.graph_nodes[self.source_index].__getitem__(sample_index)

        for i in range(0, len(self.graph_nodes)):
            if i != self.source_index:
                node = self.graph_nodes[i]
                args_node = self.graph_connectors[i][0]
                args_static = self.graph_connectors[i][1]

                args = args_static.copy()
                for arg_name in args_node:
                    op_index = args_node[arg_name][0]
                    op_output_index = args_node[arg_name][1]

                    arg = data_stack[op_index]
                    if op_output_index is not None:
                        arg = arg[op_output_index]

                    args[arg_name] = arg

                data_stack[i] = node.execute(**args)


        key_last = max(list(data_stack.keys()))
        output = data_stack[key_last]

        return output