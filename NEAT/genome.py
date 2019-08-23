
import math
import random
import pickle

import activations


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
    __input_length (int):    The amount of input nodes
    __output_length (int):   The amount of output nodes
    __ecosystem (Ecosystem): The ecosystem that this genome is a part of
    __nodes (list):          The list of nodes in the genome
    __connections (list):    The list of connections in the genome
    """

    def __init__(self, input_length, output_length, ecosystem=None):
        """
        Constructor.

        Parameters:
        input_length (int):    The amount of input nodes
        output_length (int):   The amount of output nodes
        ecosystem (Ecosystem): The ecosystem that this genome is a part of
        """
        self.__input_length = input_length
        self.__output_length = output_length
        self.__ecosystem = ecosystem
        self.__nodes = []
        self.__connections = []
        self.__fitness = 0

        # Add input and output nodes to their respective lists
        for i in range(input_length):
            self.__nodes.append(Node(len(self.__nodes), 'input'))

        for i in range(output_length):
            self.__nodes.append(Node(len(self.__nodes), 'output'))

    def __call__(self, inputs): return self.evaluate(inputs)

    def __eq__(self, genome):
        return self.__nodes == genome.get_nodes() and self.__connections == genome.get_connections()

    def __str__(self):
        """
        Represent the genome's node and connection data as a string.

        Returns:
        str: The genome's connection data
        """
        sorted_nodes = sorted(self.__nodes, key=lambda node: node.get_id())
        sorted_connections = sorted(self.__connections, key=lambda conn: conn.get_innovation_number())
        genome_string = '\n-------------Nodes-------------\n'
        for n in sorted_nodes:
            genome_string += str(n) + '\n'
        genome_string += '\n----------Connections----------\n'
        for c in sorted_connections:
            genome_string += str(c) + '\n'
        return genome_string

    def add_connection(self, node1, node2, weight=None):
        """
        Adds a connection between node1 and node2.

        Parameters:
        node1 (int): The id of the first node
        node2 (int): The id of the second node
        """
        # Make sure the nodes aren't the same
        if node1 == node2:
            raise GenomeError('Cannot connect node to itself!')

        # Check if the connection or its reverse already exist within the genome
        for c in self.__connections:
            if (c.get_in_node() == node1 and c.get_out_node() == node2) or (c.get_in_node() == node2 and c.get_out_node() == node1):
                raise GenomeError('Connection already exists within genome!')

        # Get the types of the nodes
        node1_type = self.get_node(node1).get_type()
        node2_type = self.get_node(node2).get_type()

        # Don't allow outputs to connect to outputs and inputs to inputs
        if node1_type == node2_type == 'input' or node1_type == node2_type == 'output':
            raise GenomeError('Invalid connection! Cannot connect {0} nodes!'.format(node1_type))

        # Check if the order of the nodes should be reversed
        reverse = (node1_type == 'hidden' and node2_type == 'input') or (node1_type == 'output' and node2_type == 'hidden') or (node1_type == 'output' and node2_type == 'input')

        # Make sure the connection is feed-forward
        if node2_type == 'hidden' and self.get_node_max_distance(node1) > self.get_node_max_distance(node2):
            if reverse:
                raise GenomeError('Invalid connection!  Cannot be feed-forward!')
            else:
                reverse = True

        # Create the connection and add it to the genome
        if reverse:
            inn_num = self.get_innovation_number(node2, node1)
            conn = Connection(inn_num, node2, node1, weight=weight)
        else:
            inn_num = self.get_innovation_number(node1, node2)
            conn = Connection(inn_num, node1, node2, weight=weight)

        # Add the connection to the genome
        self.__connections.append(conn)

    def add_node(self, innovation_number, activation=activations.modified_sigmoid):
        """
        Adds a hidden node onto an existing connection with the given innovation number.

        The existing connection is disabled and two new connections are added in its place.

        Parameters:
        innovation_number (int): The innovation number of the connection to add the node to
        activation (function):   The activation function for the node
        """
        # Find the connection to add the node to and disable it
        conn = self.get_connection(innovation_number)
        if conn is None:
            raise GenomeError('Connection with innovation number {0} does not exist within genome!'.format(innovation_number))
        conn.disable()

        # Create the node
        node_id = len(self.__nodes)
        self.__nodes.append(Node(node_id, 'hidden', activation=activation))

        # Create new connections to add in place of the disabled connection
        conn_id_1 = self.get_innovation_number(node_id, conn.get_out_node())
        self.__connections.append(Connection(conn_id_1, node_id, conn.get_out_node(), weight=conn.get_weight()))
        conn_id_2 = self.get_innovation_number(conn.get_in_node(), node_id)
        self.__connections.append(Connection(conn_id_2, conn.get_in_node(), node_id, weight=1.0))

    def connections_at_max(self):
        """
        Returns true if there are no possible places for new connections that are not already filled.

        Returns:
        bool: True if there are no possible places for new connections that are not already filled
        """
        input_amt = self.__input_length
        output_amt = self.__output_length
        hidden_amt = len(self.__nodes) - self.__input_length - self.__output_length

        if hidden_amt == 0:
            at_max = len(self.__connections) == input_amt * output_amt
        else:
            at_max = len(self.__connections) == (input_amt * output_amt) + (input_amt * hidden_amt) + (hidden_amt * output_amt) + (hidden_amt - 1)
        return at_max

    def copy(self):
        """
        Returns a copy of the genome.

        Returns:
        (Genome): A copy of the genome
        """
        genome_copy = Genome(self.__input_length, self.__output_length, self.__ecosystem)
        node_copies = []
        conn_copies = []
        for node in self.__nodes:
            node_copies.append(node.copy())
        for conn in self.__connections:
            conn_copies.append(conn.copy())
        genome_copy.set_nodes(node_copies)
        genome_copy.set_connections(conn_copies)
        return genome_copy

    def evaluate(self, inputs):
        """
        Evaluate the neural network (genome) and return the outputs.

        Parameters:
        inputs (list): A list of input floats of length equal to the amount of input nodes

        Returns:
        list: A list of output floats between of length equal to the amount of output nodes
        """
        if len(inputs) != self.__input_length:
            raise ValueError('Invalid input!  Amount required: {0}  Amount given: {1}'.format(self.__input_length, len(inputs)))

        # Enter the inputs and reset all non-inputs to zero
        input_index = 0
        for i in range(len(self.__nodes)):
            if self.__nodes[i].get_type() == 'input':
                self.__nodes[i].set_value(inputs[input_index])
                input_index += 1
            else:
                self.__nodes[i].set_value(0.0)

        # Calculate the outputs
        self.sort_connections()
        for conn in self.__connections:
            if conn.is_expressed():
                in_node, out_node = self.get_node(conn.get_in_node()), self.get_node(conn.get_out_node())
                out_node.set_value(out_node.get_value() + in_node.get_value() * conn.get_weight())

                # Call the activation function if the out node has finished being calculated
                if out_node.get_id() not in [c.get_out_node() for c in self.__connections[self.__connections.index(conn)+1:]]:
                    out_node.activate()

        # Return the outputs
        return [node.get_value() for node in self.__nodes if node.get_type() == 'output']

    def get_connection(self, conn):
        """
        Returns the connection with the given innovation number.

        Parameters:
        conn (int) or (tuple): The innovation number of the connection or the input and output nodes of the connection

        Returns:
        (Connection): The connection in the genome with the given innovation number/input-output nodes or None if it doesn't exist
        """
        for connection in self.__connections:
            if connection == conn:
                return connection

    def get_connections(self): return self.__connections

    def get_ecosystem(self): return self.__ecosystem

    def get_first_available_connection(self):
        """
        Returns the first potential connection that does not already exist within the genome.

        Returns:
        (tuple): The input and output node id's for an available connection, contained in a tuple
        (None):  None if no input is available
        """
        if not self.connections_at_max():
            input_nodes = [n.get_id() for n in self.__nodes if n.get_type() == 'input']
            output_nodes = [n.get_id() for n in self.__nodes if n.get_type() == 'output']
            hidden_nodes = [n.get_id() for n in self.__nodes if n.get_type() == 'hidden']

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

    def get_fitness(self): return self.__fitness

    def get_innovation_number(self, in_node, out_node):
        """
        Returns the innovation number for a connection with the given input and output.

        Parameters:
        in_node (int):  The id of the input node for the connection
        out_node (int): The id of the output node for the connection

        Returns:
        (int): The innovation number of an existing, matching connection or a new one if no matching connection exists
        """
        if self.__ecosystem is not None:
            return self.__ecosystem.get_innovation_number(in_node, out_node)
        else:
            conn = (in_node, out_node)
            if conn in self.__connections:
                return self.get_connection(conn).get_innovation_number()
            else:
                return len(self.__connections)

    def get_node(self, node_id):
        """
        Returns the node with the given id.

        Parameters:
        node_id (int): The id of the node

        Returns:
        (Node): The node in the genome with the given id or None if it doesn't exist
        """
        for node in self.__nodes:
            if node.get_id() == node_id:
                return node

    def get_node_inputs(self, node):
        """
        Returns every node that is connected to the given node as an input.

        Parameters:
        node (int): The id of the node to get the inputs for

        Returns:
        (list): A list of id's for every node that is connected to the given node as an input
        """
        node_inputs = []
        for c in self.__connections:
            if c.get_out_node() == node:
                node_inputs.append(c.get_in_node())
        return node_inputs

    def get_node_outputs(self, node):
        """
        Returns every node that is connected to the given node as an output.

        Parameters:
        node (int): The id of the node to get the outputs for

        Returns:
        (list): A list of id's for every node that is connected to the given node as an output
        """
        node_outputs = []
        for c in self.__connections:
            if c.get_in_node() == node:
                node_outputs.append(c.get_out_node())
        return node_outputs

    def get_nodes(self): return self.__nodes

    def get_node_max_distance(self, node):
        """
        Returns the maximum distance of the given node from an input node.

        Calculated recursively.

        If node does not connect to an input node, -1 is returned

        Parameters:
        node (int): The id of the node to test the distance for

        Returns:
        (int): The maximum distance from an input node
        """
        if self.get_node(node).get_type() == 'input':
            return 0

        node_inputs = self.get_node_inputs(node)

        if len(node_inputs) == 0:
            return -1

        distances = []
        for n in node_inputs:
            distances.append(self.get_node_max_distance(n) + 1)

        return max(distances)

    def mutate_add_connection(self):
        """
        Randomly adds a connection between two existing nodes.
        """
        tries = 0
        if self.connections_at_max() is False:
            while True:
                if tries < 20:
                    # Select two random, unequal nodes to connect
                    node_ids = [node.get_id() for node in self.__nodes]
                    node1 = random.choice(node_ids)
                    node2 = random.choice(node_ids)

                    try:
                        self.add_connection(node1, node2)
                        break
                    except GenomeError:
                        tries += 1
                else:
                    # Select the first available connection
                    node1, node2 = self.get_first_available_connection()

                    if node1 is not None and node2 is not None:
                        try:
                            self.add_connection(node1, node2)
                        except GenomeError:
                            pass
                    break

    def mutate_add_node(self):
        """
        Randomly adds a hidden node onto an existing connection.

        The existing connection is disabled and two new connections are added in its place.
        """
        if len(self.__connections) > 0:
            # Select a random expressed connection and disable it
            expressed_connections = [c for c in self.__connections if c.is_expressed()]
            if len(expressed_connections) == 0:
                return
            rand_conn = random.choice(expressed_connections)
            rand_conn.disable()

            # Add the node to the random connection
            self.add_node(rand_conn.get_innovation_number())

    def mutate_random(self, mutations=('add_connection', 'add_node', 'random_activation', 'random_weight', 'shift_weight', 'toggle_connection')):
        """
        Selects a random mutation and applies it to the genome.

        Parameters:
        mutations (list): A list of mutation function names to be randomly selected from (default: all of them)
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
        hidden_nodes = [node for node in self.__nodes if node.get_type() == 'hidden']
        if len(hidden_nodes) > 0:
            rand_hidden_node = random.choice(hidden_nodes)
            rand_hidden_node.set_activation(activations.get_random())

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
            rand_conn.set_weight(rand_conn.get_weight() + step)

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

        Parameters:
        path (str): The path for the new file (including the name of the file)
        """
        pickle.dump(self, open(path, 'wb'))

    def set_connections(self, connections): self.__connections = connections

    def set_ecosystem(self, ecosystem): self.__ecosystem = ecosystem

    def set_fitness(self, fitness): self.__fitness = fitness

    def set_nodes(self, nodes): self.__nodes = nodes

    def sort_connections(self):
        """
        Sorts the connections to be in feed-forward order.
        """
        i = 0
        while i < len(self.__connections):
            j = i+1
            back_amt = -1
            while j < len(self.__connections):
                if self.__connections[j].get_out_node() == self.__connections[i].get_in_node():
                    self.__connections.insert(i, self.__connections.pop(j))
                    i += 1
                    back_amt += 1
                j += 1
            i -= back_amt


class Node:
    """
    A node gene in the genome.

    Attributes:
    __id_num (int):          The node's id number within the genome
    __node_type (str):       The type of node ('input', 'output', or 'hidden')
    __activation (function): The node's activation function
    __value (float):         The current value contained in the node (the input for input nodes, the output for output
                             nodes, or the placeholder value for hidden nodes)
    """
    def __init__(self, id_num, node_type, activation=activations.modified_sigmoid, value=0.0):
        """
        Constructor.

        Parameters:
        id (int):        The node's id number within the genome
        node_type (str): The type of node ('input', 'output', or 'hidden')
        activation (function): The node's activation function
        value (float):       The current value contained in the node (the input for input nodes, the output for output
                             nodes, or the placeholder value for hidden nodes)
        """
        self.__id_num = id_num
        self.__node_type = node_type
        if node_type == 'input':
            self.__activation = None
        else:
            self.__activation = activation
        self.__value = value

    def __eq__(self, node):
        if isinstance(node, Node):
            return node.get_id() == self.__id_num
        elif isinstance(node, int):
            return node == self.__id_num
        else:
            return False

    def __str__(self):
        return '{0:3d}: {1:6s} {2}'.format(self.__id_num, self.__node_type, self.__value)

    def activate(self):
        """
        Run the node's value through its activation function.
        """
        if self.__activation is not None:
            self.__value = self.__activation(self.__value)

    def copy(self):
        """
        Returns a copy of the node.

        Returns:
        (Node): A copy of the node
        """
        return Node(self.__id_num, self.__node_type, self.__activation, self.__value)

    def get_activation(self): return self.__activation

    def get_id(self): return self.__id_num

    def get_type(self): return self.__node_type

    def get_value(self): return self.__value

    def set_activation(self, activation):
        """
        Only sets the activation for hidden and output nodes.
        """
        if self.__node_type != 'input':
            self.__activation = activation

    def set_value(self, value): self.__value = value


class Connection:
    """
    A connection gene as part of the Genome.

    Attributes:
    __innovation_number (int): The unique number associated with this connection gene that can be linked to other equal connections
    __in_node (int):           The id of the input node for the connection
    __out_node (int):          The id of the output node for the connection
    __weight (float):          The weight of the connection
    __expressed (bool):        Whether the connection is enabled or disabled
    """
    def __init__(self, innovation_number, in_node, out_node, weight=None, expressed=True):
        """
        Constructor.

        Parameters:
        innovation_number (int): The unique number associated with this connection gene that can be linked to other equal connections
        in_node (int):           The id of the input node for the connection
        out_node (int):          The id of the output node for the connection
        weight (float):          The weight of the connection
        expressed (bool):        Whether the connection is enabled or disabled
        """
        self.__innovation_number = innovation_number
        self.__in_node = in_node
        self.__out_node = out_node
        if weight is not None:
            self.__weight = weight
        else:
            self.__weight = 0.0
            self.set_random_weight()
        self.__expressed = expressed

    def __contains__(self, node):
        return self.__in_node == node or self.__out_node == node

    def __eq__(self, conn):
        if isinstance(conn, Connection):
            return conn.get_innovation_number() == self.__innovation_number
        elif isinstance(conn, int):
            return conn == self.__innovation_number
        elif isinstance(conn, tuple):
            in_node, out_node = conn
            return in_node == self.__in_node and out_node == self.__out_node
        else:
            return False

    def __str__(self):
        if self.__expressed:
            expressed = 'O'
        else:
            expressed = 'X'
        return '{0:3d}:{1}-{2} [{4}] {3}'.format(self.__innovation_number, self.__in_node, self.__out_node, self.__weight, expressed)

    def copy(self):
        """
        Returns a copy of the connection.

        Returns:
        (Connection): A copy of the connection
        """
        return Connection(self.__innovation_number, self.__in_node, self.__out_node, self.__weight, self.__expressed)

    def disable(self): self.__expressed = False

    def enable(self): self.__expressed = True

    def is_expressed(self): return self.__expressed

    def get_innovation_number(self): return self.__innovation_number

    def get_in_node(self): return self.__in_node

    def get_out_node(self): return self.__out_node

    def get_weight(self): return self.__weight

    def set_weight(self, weight): self.__weight = weight

    def set_random_weight(self): self.__weight = math.cos(random.random() * math.pi)

    def toggle(self): self.__expressed = not self.__expressed


class GenomeError(Exception):
    def __init__(self, message):
        self.message = message


def load(path):
    """
    Load a genome from the file at the given path.

    Parameters:
    The path to the file containing a pickled genome
    """
    return pickle.load(open(path, 'rb'))
