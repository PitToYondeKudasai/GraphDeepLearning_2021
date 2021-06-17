class Graph():
    # mapping = dict()

    def __init__(self):
        self.nodes = []
        self.mapping = dict()

    def add_node(self, u):
        self.nodes.append(u)
        self.mapping[u.id] = len(self.nodes) - 1

    def get_nodes(self):
        list_nodes_id = []
        # for i in Graph.mapping.values():
        #     if i != None:
        #         list_nodes_id.append(i)
        # return list_nodes_id
        for i in self.mapping.keys():
            if self.mapping[i] != None:
                list_nodes_id.append(i)
        return list_nodes_id

    def get_mapping(self):
        tmp = dict()
        super_nodes_id = self.get_nodes()
        # for i in range(len(super_nodes)):
        #     mapping[i] = self.nodes[super_nodes[i]].merged_nodes
        # return mapping
        counter = 0
        for node_id in super_nodes_id:
            tmp[counter] = self.nodes[node_id].merged_nodes
            counter += 1
        return tmp

    def add_edge(self, u_id, v_id):
        u = self.nodes[self.mapping[u_id]]
        v = self.nodes[self.mapping[v_id]]
        u.add_node(v)
        v.add_node(u)

    def remove_edge(self, u_id, v_id):
        u = self.nodes[self.mapping[u_id]]
        v = self.nodes[self.mapping[v_id]]
        u.remove_node(v)
        v.remove_node(u)

    def join_nodes(self, u_id, v_id):
        u = self.nodes[self.mapping[u_id]]
        v = self.nodes[self.mapping[v_id]]

        u.join(v)
        self.nodes[self.mapping[v_id]] = None
        self.mapping[v_id] = None

    def get_heavy_edge(self, u_id):
        try:
            _ = self.mapping[u_id]
        except:
            print("errore 1. u_id: ", u_id, "mapping: ",self.mapping)

        try:
            u = self.nodes[self.mapping[u_id]]
        except Exception as e:
            print(u_id)
            print(self.nodes)
            raise e

        heavy_node_id = None
        heavy_edge = float('inf')
        for v in u.neigh:
            if  v.degree < heavy_edge:
                heavy_edge = v.degree
                heavy_node_id = v.id
        return heavy_node_id


class Node():
    def __init__(self, i):
        self.neigh = []
        self.degree = 0
        self.id = i
        self.merged_nodes = [i]

    def join(self, v):
        self.merged_nodes.append(v.id)
        for node in v.neigh:
            node.remove_node(v)
            node.add_node(self)
            self.add_node(node)

    def add_node(self, u):
        if not(u in self.neigh) and (u.id != self.id):
            self.neigh.append(u)
            self.degree += 1

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def remove_node(self, u):
        try:
            self.neigh.remove(u)
            self.degree -= 1
        except:
            pass
