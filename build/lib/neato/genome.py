"""
Neato

MIT License

Copyright (c) 2021 Joseph Pace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import itertools
import math
import random
import pickle

import numpy as np

from . import activations


class Genome:
    """
    A genome is an object containing the genetic information of an evolved neural network.

    Each genome can have anywhere between none and several connections between input nodes,
    hidden nodes, and output nodes.

    Each genome contains a fixed amount of input nodes and output nodes. Hidden nodes can
    be added at random existing connections.

    Genomes can be mutated to add nodes, add connections, change connection weights, and
    toggle connection expression.

    Attributes:
        __connections (list):    The list of connections in the genome.
        __nodes (list):        The list of nodes in the genome.
        __order (list):        The order that nodes will be evaluated.
        ecosystem (Ecosystem): The ecosystem that this genome is a part of.
        shape (tuple):         The amount of input and output nodes.
        weights (ndarray):     The weights for each connection between each node.
    """

    def __init__(self, input_size, output_size, ecosystem=None):
        """
        Constructor.

        Args:
            input_size (int):      The amount of input nodes.
            output_size (int):     The amount of output nodes.
            ecosystem (Ecosystem): The ecosystem that this genome is a part of.
        """
        self.shape = (input_size, output_size)
        self.ecosystem = ecosystem

        self.__nodes = []
        self.__connections = []
        self.fitness = 0

        # Add input and output nodes to their respective lists
        for i in range(input_size):
            self.__nodes.append(Node(len(self.__nodes), 'input'))

        for i in range(output_size):
            self.__nodes.append(Node(len(self.__nodes), 'output'))

        # Compile weights
        self.__compile()

    def __call__(self, x): return self.forward(x)

    def __eq__(self, genome):
        return self.__nodes == genome.get_nodes() and self.__connections == genome.get_connections()

    def __str__(self):
        """
        Represent the genome's node and connection data as a string.

        Returns:
            str: The genome's connection data.
        """
        sorted_nodes = sorted(self.__nodes, key=lambda node: node.id)
        sorted_connections = sorted(self.__connections, key=lambda conn: conn.innovation_number)
        genome_string = '\nNodes:\n'
        for n in sorted_nodes:
            genome_string += f'\t{str(n)}\n'
        genome_string += '\nConnections:\n'
        for c in sorted_connections:
            genome_string += f'\t{str(c)}\n'
        return genome_string

    def __compile(self):
        """
        Compiles the weights of each connection into an ndarray for faster evaluation in the forward pass.
        """
        self.__order = []
        nodes = []
        for node in self.__nodes:
            if node.type != 'input':
                nodes.append(node)
        nodes.sort(key=lambda n: self.get_node_max_distance(n.id))
        self.weights = np.zeros((len(nodes), len(self.__nodes)))
        for i, out_node in enumerate(nodes):
            self.__order.append(out_node.id)
            for in_node in self.__nodes:
                conn = self.get_connection_from_nodes(in_node.id, out_node.id)
                if conn is not None and conn.expressed:
                    self.weights[i, in_node.id] = conn.weight

    def add_connection(self, nid1, nid2, weight=None):
        """
        Adds a connection between the nodes with the given ids.

        Args:
            nid1 (int):    The id of the first node.
            nid2 (int):    The id of the second node.
            weight (float): The weight of the connection.
        """
        # Get the nodes
        node1 = self.get_node(nid1)
        node2 = self.get_node(nid2)

        # Make sure the nodes exist
        assert node1 is not None, f'Node {nid1} does not exist within genome!'
        assert node2 is not None, f'Node {nid2} does not exist within genome!'

        # Make sure the connection does not already exist within the genome
        for c in self.__connections:
            assert not (c.in_node == nid1 and c.out_node == nid2), 'Connection already exists within genome!'

        # Make sure the connection is valid
        assert nid1 != nid2, f'Invalid connection! Cannot connect node to self!'
        if node1.type != 'hidden':
            assert node1.type != node2.type, f'Invalid connection! {node1.type} -> {node2.type}'

        # Check if the order of the nodes should be reversed
        reverse = (node1.type == 'hidden' and node2.type == 'input') or (node1.type == 'output' and node2.type == 'hidden') or (node1.type == 'output' and node2.type == 'input')

        # Create the connection and add it to the genome
        if reverse:
            inn_num = self.get_innovation_number(nid2, nid1)
            conn = Connection(inn_num, nid2, nid1, weight=weight)
        else:
            inn_num = self.get_innovation_number(nid1, nid2)
            conn = Connection(inn_num, nid1, nid2, weight=weight)

        # Add the connection to the genome and compile weights
        self.__connections.append(conn)
        self.__compile()

    def add_node(self, innovation_number, activation=activations.modified_sigmoid):
        """
        Adds a hidden node onto an existing connection with the given innovation number.

        The existing connection is disabled and two new connections are added in its place.

        Args:
            innovation_number (int): The innovation number of the connection to add the node to.
            activation (function):   The activation function for the node.
        """
        # Find the connection to add the node to and disable it
        conn = self.get_connection(innovation_number)
        assert conn is not None, f'Connection with innovation number {innovation_number} does not exist within genome!'
        conn.disable()

        # Create the node
        node_id = len(self.__nodes)
        self.__nodes.append(Node(node_id, 'hidden', activation=activation))

        # Create new connections to add in place of the disabled connection
        conn_id_1 = self.get_innovation_number(node_id, conn.out_node)
        self.__connections.append(Connection(conn_id_1, node_id, conn.out_node, weight=conn.weight))
        conn_id_2 = self.get_innovation_number(conn.in_node, node_id)
        self.__connections.append(Connection(conn_id_2, conn.in_node, node_id, weight=1.0))

        # Compile weights
        self.__compile()

    def connections_at_max(self):
        """
        Returns true if there are no possible places for new connections that are not already filled.

        Returns:
            bool: True if there are no possible places for new connections that are not already filled.
        """
        input_size, output_size = self.shape
        input_amt = input_size
        output_amt = output_size
        hidden_amt = len(self.__nodes) - input_size - output_size

        if hidden_amt == 0:
            at_max = len(self.__connections) == input_amt * output_amt
        else:
            at_max = len(self.__connections) == (input_amt * output_amt) + (input_amt * hidden_amt) + (hidden_amt * output_amt) + (hidden_amt - 1)
        return at_max

    def copy(self):
        """
        Returns a copy of the genome.

        Returns:
            (Genome): A copy of the genome.
        """
        input_size, output_size = self.shape
        genome_copy = Genome(input_size, output_size, self.ecosystem)
        node_copies = [n.copy() for n in self.__nodes]
        conn_copies = [c.copy() for c in self.__connections]
        genome_copy.set_nodes(node_copies)
        genome_copy.set_connections(conn_copies)
        genome_copy.weights = self.weights
        return genome_copy

    def forward(self, x):
        """
        The forward pass for the genome's neural network.

        Args:
            x (ndarray): The inputs to the network.

        Returns:
            (ndarray): The outputs of the network.
        """
        assert len(x) == self.shape[0], f'Invalid input size!  Amount required: {len(x)}  Amount given: {self.shape[0]}'
        y = np.pad(x, (0, len(self.__nodes) - len(x)))
        for nid, w in zip(self.__order, self.weights):
            node = self.get_node(nid)
            y[nid] += y.dot(w)
            y[nid] = node.activation(y[nid]) if w.any() else y[nid]
        input_size, output_size = self.shape
        y = y[input_size:input_size + output_size]
        return y

    def get_connection(self, conn):
        """
        Returns the connection with the given innovation number.

        Args:
            conn (int) or (tuple): The innovation number of the connection or the input and output nodes of the connection.

        Returns:
            (Connection): The connection with the given innovation number/input-output nodes or None if it doesn't exist.
        """
        for connection in self.__connections:
            if connection == conn:
                return connection

    def get_connections(self): return self.__connections

    def get_connection_from_nodes(self, in_node, out_node):
        """
        Returns the connection with the given in_node and out_node.

        Args:
            in_node (int):  The id of the in_node for the connection.
            out_node (int): The id of the out_node for the connection.

        Returns:
            (Connection): The connection with the given in_node and out_node.
        """
        for connection in self.__connections:
            if connection.in_node == in_node and connection.out_node == out_node:
                return connection

    def get_first_available_connection(self):
        """
        Returns the first potential connection that does not already exist within the genome.

        Returns:
            (tuple): The input and output node id's for an available connection, contained in a tuple.
            (None):  None if no input is available.
        """
        if not self.__connections_at_max():
            input_nodes = [n.id for n in self.__nodes if n.type == 'input']
            output_nodes = [n.id for n in self.__nodes if n.type == 'output']
            hidden_nodes = [n.id for n in self.__nodes if n.type == 'hidden']

            for i in input_nodes:
                for o in output_nodes:
                    if (i, o) not in self.__connections:
                        return i, o

            for h in hidden_nodes:
                for o in output_nodes:
                    if (h, o) not in self.__connections:
                        return h, o

            for i in input_nodes:
                for h in hidden_nodes:
                    if (i, h) not in self.__connections:
                        return i, h

            for h1 in hidden_nodes:
                for h2 in hidden_nodes:
                    if h1 == h2:
                        continue
                    elif (h1, h2) not in self.__connections:
                        return h1, h2
        return None, None

    def get_innovation_number(self, in_node, out_node):
        """
        Returns the innovation number for a connection with the given input and output.

        Args:
            in_node (int):  The id of the input node for the connection.
            out_node (int): The id of the output node for the connection.

        Returns:
            (int): The innovation number of an existing, matching connection or a new one if no matching connection exists.
        """
        if self.ecosystem is not None:
            return self.ecosystem.get_innovation_number(in_node, out_node)
        else:
            conn = (in_node, out_node)
            if conn in self.__connections:
                return self.get_connection(conn).innovation_number
            else:
                return len(self.__connections)

    def get_node(self, node_id):
        """
        Returns the node with the given id.

        Args:
            node_id (int): The id of the node.

        Returns:
            (Node): The node in the genome with the given id or None if it doesn't exist.
        """
        for node in self.__nodes:
            if node.id == node_id:
                return node

    def get_node_inputs(self, node):
        """
        Returns every node that is connected to the given node as an input.

        Args:
            node (int): The id of the node to get the inputs for.

        Returns:
            (list): A list of id's for every node that is connected to the given node as an input.
        """
        node_inputs = []
        for c in self.__connections:
            if c.out_node == node:
                node_inputs.append(c.in_node)
        return node_inputs

    def get_node_max_distance(self, node, visited=None):
        """
        Returns the maximum distance of the given node from an input node.
        Calculated recursively.

        If node does not connect to an input node, -1 is returned

        Args:
            node (int):    The id of the node to test the distance for
            visited (set): The set of visited nodes on the current recursion path.

        Returns:
            (int): The maximum distance from an input node
        """
        # Input nodes have a distance of 0
        if self.get_node(node).type == 'input':
            return 0

        # Only visit each node once to avoid recursion error
        visited = set() if visited is None else visited.copy()
        if node in visited:
            return 0
        visited.add(node)

        # Get node inputs
        node_inputs = self.get_node_inputs(node)
        if len(node_inputs) == 0:
            return -1

        # Find distances of all inputs
        distances = []
        for n in node_inputs:
            distances.append(self.get_node_max_distance(n, visited=visited) + 1)

        # Return max distance
        return max(distances)

    def get_node_outputs(self, node):
        """
        Returns every node that is connected to the given node as an output.

        Args:
            node (int): The id of the node to get the outputs for.

        Returns:
            (list): A list of id's for every node that is connected to the given node as an output.
        """
        node_outputs = []
        for c in self.__connections:
            if c.in_node == node:
                node_outputs.append(c.out_node)
        return node_outputs

    def get_nodes(self): return self.__nodes

    def mutate_add_connection(self):
        """
        Randomly adds a connection between two existing nodes.
        """
        # Get current connections
        connections = set()
        for conn in self.__connections:
            connections.add((conn.in_node, conn.out_node))

        # Get node info
        nodes = {'all': [], 'input': [], 'hidden': [], 'output': []}
        for node in self.__nodes:
            nodes['all'].append(node.id)
            nodes[node.type].append(node.id)

        # Find all possible connection choices
        invalid_choices = []
        invalid_combos = [('output', 'input'), ('hidden', 'input'), ('input', 'input'), ('output', 'output')]
        for in_type, out_type in invalid_combos:
            invalid_choices += list(itertools.product(nodes[in_type], nodes[out_type]))
        invalid_choices += list(zip(nodes['all'], nodes['all']))  # Exclude self to self
        invalid_choices = set(invalid_choices).union(connections)  # Exclude current connections
        choices = set(itertools.product(nodes['all'], nodes['all'])).difference(invalid_choices)

        # Pick a random connection and add it to the rest
        if len(choices) > 0:
            node1, node2 = random.choice(list(choices))
            self.add_connection(node1, node2)

    def mutate_add_node(self):
        """
        Randomly adds a hidden node onto an existing connection. If no connection is present, no node will be added.

        The existing connection is disabled and two new connections are added in its place.
        """
        if len(self.__connections) > 0:
            # Select a random expressed connection and disable it
            expressed_connections = [c for c in self.__connections if c.expressed]
            if len(expressed_connections) == 0:
                return
            rand_conn = random.choice(expressed_connections)
            rand_conn.disable()

            # Add the node to the random connection
            self.add_node(rand_conn.innovation_number)

    def mutate_random(self, mutations=('add_connection', 'add_node', 'random_activation', 'random_weight', 'shift_weight', 'toggle_connection')):
        """
        Selects a random mutation and applies it to the genome.

        Args:
            mutations (list): A list of mutation function names to be randomly selected from (default: all of them).
        """
        mutation_functions = {
            'add_connection': self.mutate_add_connection,
            'add_node': self.mutate_add_node,
            'random_activation': self.mutate_random_activation,
            'random_weight': self.mutate_random_weight,
            'shift_weight': self.mutate_shift_weight,
            'toggle_connection': self.mutate_toggle_connection
        }
        mutation_functions[random.choice(mutations)]()

    def mutate_random_activation(self):
        """
        Randomly selects a hidden node and sets its activation to a random function.
        """
        hidden_nodes = [node for node in self.__nodes if node.type == 'hidden']
        if len(hidden_nodes) > 0:
            rand_hidden_node = random.choice(hidden_nodes)
            rand_hidden_node.activation = activations.get_random()

    def mutate_random_weight(self):
        """
        Randomly selects a connection and sets its weight to a random value.
        """
        if len(self.__connections) > 0:
            rand_conn = random.choice(self.__connections)
            rand_conn.set_random_weight()

    def mutate_shift_weight(self, step=0.1):
        """
        Randomly selects a connection and shifts its weight up or down a small step.
        """
        if len(self.__connections) > 0:
            # Randomly select whether it increases or decreases by the step
            random_sign = 1 if math.cos(random.random() * math.pi) > 0 else -1
            step *= random_sign

            # Select a random connection and shift its weight
            rand_conn = random.choice(self.__connections)
            rand_conn.weight = rand_conn.weight + step

    def mutate_toggle_connection(self):
        """
        Randomly selects a connection and toggles its expression.

        If it's enabled, it will become disabled and vice versa.
        """
        if len(self.__connections) > 0:
            # Select a random connection and toggle it
            rand_conn = random.choice(self.__connections)
            rand_conn.toggle()

    def save(self, path):
        """
        Pickles the genome and saves it to the given path.

        If only the name is given, the genome will be saved to the directory of the file that created the genome.

        Args:
            path (str): The path for the new file (including the name of the file).
        """
        pickle.dump(self, open(path, 'wb'))

    def set_activation(self, node_id, activation):
        """
        Sets the activation function for the node with the given id to the given function.

        Args:
            node_id (int):         The node id.
            activation (function): The activation function.
        """
        node = self.get_node(node_id)
        if node is not None:
            node.activation = activation

    def set_connections(self, connections):
        self.__connections = connections
        self.__compile()

    def set_nodes(self, nodes):
        self.__nodes = nodes
        input_size = 0
        output_size = 0
        for node in nodes:
            if node.type == 'input':
                input_size += 1
            elif node.type == 'output':
                output_size += 1
        self.shape = (input_size, output_size)
        self.__compile()


class Node:
    """
    A node gene in the genome.

    Attributes:
        activation (function): The node's activation function.
        id (int):              The node's id number within the genome.
        type (str):            The type of node ('input', 'output', or 'hidden').
    """
    def __init__(self, id_num, node_type, activation=None):
        """
        Constructor.

        Args:
            id_num (int):          The node's id number within the genome.
            node_type (str):       The type of node ('input', 'output', or 'hidden').
            activation (function): The node's activation function.
        """
        self.id = id_num
        self.type = node_type
        if activation is None:
            self.activation = activations.linear if node_type == 'input' else activations.modified_sigmoid
        else:
            self.activation = activation

    def __eq__(self, node):
        if isinstance(node, Node):
            return node.id == self.id
        elif isinstance(node, int):
            return node == self.id
        else:
            return False

    def __str__(self):
        return f'{self.id:3d}: {self.type:6s}'

    def copy(self):
        """
        Returns a copy of the node.

        Returns:
            (Node): A copy of the node.
        """
        return Node(self.id, self.type, activation=self.activation)


class Connection:
    """
    A connection gene as part of the Genome.

    Attributes:
        expressed (bool):        Whether the connection is enabled or disabled.
        in_node (int):           The id of the input node for the connection.
        innovation_number (int): The unique number associated with this connection gene that can be linked to other equal connections.
        out_node (int):          The id of the output node for the connection.
        weight (float):          The weight of the connection.
    """
    def __init__(self, innovation_number, in_node, out_node, weight=None, expressed=True):
        """
        Constructor.

        Args:
            innovation_number (int): The unique number associated with this connection gene that can be linked to other equal connections.
            in_node (int):           The id of the input node for the connection.
            out_node (int):          The id of the output node for the connection.
            weight (float):          The weight of the connection.
            expressed (bool):        Whether the connection is enabled or disabled.
        """
        self.innovation_number = innovation_number
        self.in_node = in_node
        self.out_node = out_node
        self.expressed = expressed
        self.weight = math.cos(random.random() * math.pi) if weight is None else weight

    def __contains__(self, node):
        return self.in_node == node or self.out_node == node

    def __eq__(self, conn):
        if isinstance(conn, Connection):
            return conn.innovation_number == self.innovation_number
        elif isinstance(conn, int):
            return conn == self.innovation_number
        elif isinstance(conn, tuple):
            in_node, out_node = conn
            return in_node == self.in_node and out_node == self.out_node
        else:
            return False

    def __str__(self):
        if self.expressed:
            expressed = 'O'
        else:
            expressed = 'X'
        return f'{self.innovation_number:3d}:{self.in_node}-{self.out_node} [{expressed}] {self.weight}'

    def copy(self):
        """
        Returns a copy of the connection.

        Returns:
            (Connection): A copy of the connection.
        """
        return Connection(self.innovation_number, self.in_node, self.out_node, self.weight, self.expressed)

    def disable(self): self.expressed = False

    def enable(self): self.expressed = True

    def set_random_weight(self): self.weight = math.cos(random.random() * math.pi)

    def toggle(self): self.expressed = not self.expressed


def load(path):
    """
    Load a genome from the file at the given path.

    Args:
        path (str): The path to the file containing a pickled genome.
    """
    return pickle.load(open(path, 'rb'))
