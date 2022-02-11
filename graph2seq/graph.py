import numpy as np
from abc import ABC, abstractmethod


class Graph(ABC):
    def __init__(self, G):
        '''
        Args:
            G(list): a graph
        '''
        self.G = np.array(G)

    @abstractmethod
    def nodes(self):
        pass

    @abstractmethod
    def add_edges(self, edge):
        pass

    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def neighbors(self, node):
        pass

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def has_node(self, node):
        pass

    @abstractmethod
    def has_edge(self, node, neighbor):
        pass


class HomogeneousGraph(Graph):

    def nodes(self):
        return np.unique(np.concatenate([self.G.T[0], self.G.T[1]]))

    def edges(self):
        return np.unique(self.G, axis=0)

    def weights(self, node=None, neighbor=None):
        if not any([node, neighbor]):
            condition = ()
        elif all([node, neighbor]):
            condition = (self.G.T[0] == node) & (self.G.T[1] == neighbor)
        elif not neighbor:
            condition = (self.G.T[0] == node)
        else:
            condition = (self.G.T[1] == neighbor)

        unique, counts = np.unique(self.G[condition], return_counts=True, axis=0)
        return counts.sum()

    def neighbors(self, node):
        return np.unique(self.G[(self.G.T[0] == node)].T[1])

    def has_edge(self, node, neighbor):
        return self.G[(self.G.T[0] == node) & (self.G.T[1] == neighbor)].shape[0] > 0

    def has_node(self, node):
        return node in self.nodes()

    def add_edges(self, edge):
        self.G = np.append(self.G, [edge], axis=0)

    def __str__(self):
        return self.G.__str__()

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(self, id_, type_=None, desc_=None):
        self.id_ = id_
        self.type_ = type_
        self.desc_ = desc_

    def __str__(self):
        return self.id_

    def __eq__(self, node):
        return self.id_ == node.id_ and self.type_ == node.type_

    def __gt__(self, node):
        return self.id_ > node.id_

    def __repr__(self):
        return self.id_

    def __hash__(self):
        return hash(str(self))


class Edge:
    def __init__(self, node, neighbor, id_=None, type_=None, desc_=None):
        self.id_ = id_ if id_ is not None else 'None'
        self.type_ = type_
        self.desc_ = desc_
        self.node, self.neighbor = node, neighbor

    def __eq__(self, edge):
        return self.type_ == edge.type_ and (self.node == edge.node and self.neighbor == edge.neighbor or
                                             self.node == edge.neighbor and self.neighbor == edge.node)

    def __gt__(self, edge):
        return self.id_ > edge.id_

    def __str__(self):
        return f'{str(self.node)}-{self.type_}-{str(self.neighbor)}'

    def __repr__(self):
        return f'{str(self.node)}-{self.type_}-{str(self.neighbor)}'

    def __hash__(self):
        return hash(str(self))


class HeterogeneousGraph(Graph):
    def nodes(self):
        return np.unique(np.concatenate([self.G.T[0], self.G.T[2]]))

    def add_edges(self, edge):
        self.G = np.append(self.G, [edge], axis=0)

    def edges(self, node=None, neighbor=None):
        if not any([node, neighbor]):
            condition = ()
        elif all([node, neighbor]):
            condition = (self.G.T[0] == node) & (self.G.T[2] == neighbor)
        elif not node:
            condition = (self.G.T[2] == neighbor)
        else:
            condition = (self.G.T[0] == node)

        edges_ = np.delete(self.G[condition], [0, 2], axis=1)
        np.array(list({i[0] for i in edges_}))
        # np.unique yields some issues with custom classes
        #return np.unique(np.delete(G.G, [0, 2], axis=1).T)
        return np.array(list({i[0] for i in edges_}))

    def neighbors(self, node, edge=None):
        if edge is None:
            condition = (self.G.T[0] == node)
        else:
            condition = (self.G.T[0] == node) & (np.isin(self.G.T[1], edge))

        return self.G[condition].T[1:].T

    def weights(self, node=None, neighbor=None, edge=None):
        if not any([node, neighbor, edge]):
            condition = ()
        elif all([node, neighbor, edge]):
            condition = (self.G.T[0] == node) & (self.G.T[1] == edge) & (self.G.T[2] == neighbor)
        elif all([node, neighbor]) and not edge:
            condition = (self.G.T[0] == node) & (self.G.T[2] == neighbor)
        elif all([node, edge]) and not neighbor:
            condition = (self.G.T[0] == node) & (self.G.T[1] == edge)
        elif all([neighbor, edge]) and not node:
            condition = (self.G.T[2] == neighbor) & (self.G.T[1] == edge)
        elif not any([node, neighbor]) and edge:
            condition = (self.G.T[1] == edge)
        elif not any([node, edge]) and neighbor:
            condition = (self.G.T[2] == neighbor)
        elif not any([neighbor, edge]) and node:
            condition = (self.G.T[0] == node)

        return np.count_nonzero(self.G[condition])

    def has_edge(self, node, neighbor=None, edge=None):
        if all([neighbor, edge]):
            condition = (self.G.T[0] == node) & (self.G.T[1] == edge) & (self.G.T[2] == neighbor)
        elif not any([neighbor, edge]):
            condition = (self.G.T[0] == node)
        elif not edge:
            condition = (self.G.T[0] == node) & (self.G.T[2] == neighbor)
        else:
            condition = (self.G.T[0] == node) & (self.G.T[1] == edge)

        return self.G[condition].shape[0] > 0

    def has_node(self, node):
        return node in self.nodes()
