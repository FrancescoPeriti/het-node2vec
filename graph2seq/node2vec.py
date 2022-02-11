import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from abc import ABC, abstractmethod
from graph2seq.alias_sampler import AliasSampler
from graph2seq.graph import Graph, HeterogeneousGraph, HomogeneousGraph

class node2vec(ABC):
    '''
    Args:
        G(Graph): a Graph
        p(float, optional, default=1): return parameter
        q(float, optional, default=2): walk away parameter
        gamma(int, optional, default=10): walks per vertex
        length(int, optional, default=10): walk length
    '''

    @abstractmethod
    def __init__(self, G:Graph, p:float=1, q:float=1, gamma:int=10, length:int=15, workers:int=10):
        '''
        Args:
            G(Graph): input graph
            p(float): return parameter
            q(float): in-out parameter
            gamma(int): number of walks per nodes
            length(int): walk length
            workers(int): num of processes
        '''
        self.G = G
        self.gamma, self.length = gamma, length
        self.p, self.q = p, q

        self._workers = workers

        self._preprocess_transition_probs()

    def __walks(self):
        walks = list()
        for i in range(0, self.gamma):
            nodes = self.G.nodes()

            # uncomment to shuffle nodes: it may be slow
            # np.random.shuffle(nodes)

            for node in nodes:
                walk = self.biased_random_walk(node)
                walks.append(walk)

        return walks

    def walks(self) -> list:
        walks = list()

        with Pool(processes=self._workers) as p:
            for i in tqdm(range(0, self.gamma)):
                nodes = self.G.nodes()

                # uncomment to shuffle nodes: it may be slow
                #np.random.shuffle(nodes)

                walks_i = p.map(self.biased_random_walk, nodes)
                walks.extend(walks_i)

            return walks

    @abstractmethod
    def biased_random_walk(self, node) -> list:
        pass

    @abstractmethod
    def _preprocess_transition_probs(self):
        pass

class Node2Vec(node2vec):
    '''A class to perform 2-order random walks on a Homogeneous Graph in order to derive the data needed for the
    node2vec algorithm.'''

    def __init__(self, G:HomogeneousGraph, p:float=1, q:float=2, gamma=10, length=15, workers:int=10):
        '''
        Args:
            G(HomogeneousGraph): a Graph
            p(float): return parameter
            q(float): walk away parameter
            gamma(int, optional, default=10): walks per vertex
            length(int, optional, default=15): walk length
            workers(int, optional, default=10): number of processes
        '''
        self.__sampler = AliasSampler()

        super().__init__(G, p, q, gamma, length, workers)

    def _preprocess_transition_probs(self):
        '''Preprocessing of transition probabilities for guiding the random walks.'''

        # Pool.map results are ordered.
        # If you need order, great; if you don't, Pool.imap_unordered may be a useful optimization.
        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary.
        self.__alias_nodes = dict()
        with Pool(processes=self._workers) as p:
            for input_node, alias in p.map(self._get_alias_node, self.G.nodes()):
                self.__alias_nodes[input_node] = alias

        self.__alias_edges = dict()
        with Pool(processes=self._workers) as p:
            for input_edge, alias in p.map(self._get_alias_edge, self.G.edges()):
                self.__alias_edges[input_edge] = alias

        # version without multiprocessing
        #self.__alias_nodes = dict()
        #for curr in self.G.nodes():
        #    self.__alias_nodes[curr] = self.__get_alias_node(curr)

        #self.__alias_edges = dict()
        #for edge in self.G.edges():
        #    prev, curr = tuple(edge)
        #    self.__alias_edges[(prev, curr)] = self.__get_alias_edge(prev, curr)

    def random_walk(self, node) -> list:
        walk = [node]

        # random walk of 'length' step
        for i in range(0, self.length - 1):
            # unnormalized probs
            probs = np.array([self.G.weights(node=node, neighbor=nbr) for nbr in self.G.neighbors(node)])

            # normalization factor
            norm = probs.sum()

            # normalized probs
            n_probs = probs / norm

            node = np.random.choice(self.G.neighbors(node), p=n_probs)
            walk.append(node)

        return walk

    def biased_random_walk(self, node) -> list:
        walk = [node]

        # random walk of 'length' step
        for i in range(0, self.length - 1):
            # current node
            curr = walk[-1]
            # neighbors
            nbrs = self.G.neighbors(curr)
            if nbrs.shape[0] > 0:
                if len(walk) == 1:
                    J, q = self.__alias_nodes[curr]
                    index = self.__sampler.alias_draw(J, q)
                    walk.append(nbrs[index])
                else:
                    prev = walk[-2]
                    J, q = self.__alias_edges[(prev, curr)]
                    index = self.__sampler.alias_draw(J, q)
                    walk.append(nbrs[index])
            else:
                break

        return walk

    def _get_alias_edge(self, prev_curr):
        '''Get the alias edge setup lists for a given edge.'''

        prev, curr = prev_curr

        # unnormalized probs
        probs = list()
        for nbr in self.G.neighbors(curr):
            w = self.G.weights(node=curr, neighbor=nbr)

            if nbr == prev:
                probs.append(w / self.p)
            elif self.G.has_edge(nbr, prev):
                probs.append(w)
            else:
                probs.append(w / self.q)
        probs = np.array(probs)

        # normalization
        norm = probs.sum()

        # normalized probs
        n_probs = probs / norm

        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary. We return a pair (input, output) in order to
        # be aware of the order
        return [(prev, curr), self.__sampler.alias_setup(n_probs)]
        # version without multiprocessing:
        # return self.__sampler.alias_setup(n_probs)

    def _get_alias_node(self, node):

        # unnormalized probs
        probs = np.array([self.G.weights(node=node, neighbor=nbr) for nbr in self.G.neighbors(node)])

        # normalization
        norm = probs.sum()

        # normalized probs
        n_probs = probs / norm

        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary. We return a pair (input, output) in order to
        # be aware of the order
        return [node, self.__sampler.alias_setup(n_probs)]
        # version without multiprocessing
        # return self.__sampler.alias_setup(n_probs)

class HetNode2Vec(node2vec):
    '''A class to perform 2-order random walks on a Heterogeneous Graph in order to derive the data needed for the
    Het-node2vec algorithm.'''

    def __init__(self, G:HeterogeneousGraph, p:float=1, q:float=2, s:float=0.9, e:float=0.8, gamma:int=10,
                 length:int=15, workers:int=10):
        '''
        Args:
            G(Graph): a Graph
            p(float, optional, default=1): return parameter
            q(float, optional, default=2): walk away parameter
            s(float, optional, default=0.9): switching nodes
            e(float, optional, default=0.8): switching edge
            gamma(int, optional, default=10): walks per vertex
            length(int, optional, default=10): walk length
            workers(int, optional, default=10): number of processes
        '''

        self.s, self.e = s, e
        self.__sampler = AliasSampler()

        super().__init__(G, p, q, gamma, length, workers)

    def _preprocess_transition_probs(self):
        '''Preprocessing of transition probabilities for guiding the random walks.'''

        # Pool.map results are ordered.
        # If you need order, great; if you don't, Pool.imap_unordered may be a useful optimization.
        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary.
        self.__alias_nodes = dict()
        with Pool(processes=self._workers) as p:
            for input_node, alias in p.map(self._get_alias_node, self.G.nodes()):
                self.__alias_nodes[input_node] = alias

        self.__alias_edges = dict()
        with Pool(processes=self._workers) as p:
            for input_edge, alias in p.map(self._get_alias_edge, self.G.edges()):
                self.__alias_edges[input_edge] = alias

        # version without multiprocessing
        # self.__alias_nodes = dict()
        # for curr in self.G.nodes():
        #    self.__alias_nodes[curr] = self.__get_alias_node(curr)

        # self.__alias_edges = dict()
        #         for edge in self.G.edges():
        #             self.__alias_edges[edge] = self.__get_alias_edge(edge)

    def biased_random_walk(self, node):
        walk = [node]
        link = []

        # random walk of 'length' step
        for i in range(0, self.length - 1):
            # current node
            curr = walk[-1]

            # edge to neighbors and neighbors
            edges, nbrs = self.G.neighbors(curr).T

            if nbrs.shape[0] > 0:
                if len(walk) == 1:
                    J, q = self.__alias_nodes[curr]
                    index = self.__sampler.alias_draw(J, q)
                    walk.append(nbrs[index])
                    link.append(edges[index])
                else:
                    J, q = self.__alias_edges[link[-1]]
                    index = self.__sampler.alias_draw(J, q)
                    walk.append(nbrs[index])
                    link.append(edges[index])
            else:
                break

        return walk

    def _get_alias_edge(self, edge):
        '''Get the alias edge setup lists for a given edge.'''

        prev, curr = edge.node, edge.neighbor

        # unnormalized probs
        probs = list()
        for edge_nbr, nbr in self.G.neighbors(curr):
            w = self.G.weights(node=curr, neighbor=nbr, edge=edge_nbr)

            # d_r,x = 0 --> coming back
            if nbr == prev:
                prob = w / self.p

            # d_r,x = 1
            elif self.G.has_edge(nbr, prev):
                prob = w

            # d_r,x = 2 --> going ahead
            else:
                prob = w / self.q

            # node switching
            if nbr.type_ != curr.type_:
                prob = prob / self.s

            # edge switching
            if edge_nbr.type_ != edge.type_:
                prob = prob / self.e

            probs.append(prob)

        probs = np.array(probs)

        # normalization
        norm = probs.sum()

        # normalized probs
        n_probs = probs / norm

        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary. We return a pair (input, output) in order to
        # be aware of the order
        return [edge, self.__sampler.alias_setup(n_probs)]
        # version without multiprocessing:
        # return self.__sampler.alias_setup(n_probs)

    def _get_alias_node(self, node):

        # unnormalized probs
        probs = np.array([self.G.weights(node=node, neighbor=nbr, edge=edge) for edge, nbr in self.G.neighbors(node)])

        # normalization
        norm = probs.sum()

        # normalized probs
        n_probs = probs / norm

        # Note that while the order in which you receive the results from Pool.map is fixed,
        # the order in which they are computed is arbitrary. We return a pair (input, output) in order to
        # be aware of the order
        return [node, self.__sampler.alias_setup(n_probs)]
        # version without multiprocessing
        # return self.__sampler.alias_setup(n_probs)


if __name__ == '__main__':
    from random import randrange
    G = HomogeneousGraph([[randrange(10), randrange(10)] for i in range(10000)])
    print(Node2Vec(G).biased_random_walk(1))

    from graph2seq.graph import Node, Edge
    edges = list()
    for i in range(2000):
        n1, n2 = Node(str(randrange(70))), Node(str(randrange(70)))
        e = Edge(n1, n2, id_=str(randrange(70)), type_=str(randrange(70)))
        edges.append([n1, e, n2])

    G = HeterogeneousGraph(edges)
    print(HetNode2Vec(G).biased_random_walk(Node(str(1))))
    import dill
