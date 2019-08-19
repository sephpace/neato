
import math
import random

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
    __input_length (int):            The amount of input nodes
    __output_length (int):           The amount of output nodes
    __inn_num_gen (generator):       The innovation number generator used to assign innovation numbers to connections
    __nodes (list):                  The list of nodes in the genome
    __connections (list):            The list of connections in the genome
    __node_id_index (int):           An id that is incremented with each new node to ensure uniqueness
    """

    def __init__(self, input_length, output_length, inn_num_gen):
        """
        Constructor.

        Parameters:
        input_length (int):            The amount of input nodes
        output_length (int):           The amount of output nodes
        inn_num_gen (generator):       The innovation number generator used to assign innovation numbers to connections
        """
        self.__input_length = input_length
        self.__output_length = output_length
        self.__inn_num_gen = inn_num_gen
        self.__nodes = []
        self.__connections = []
        self.__node_id_index = 0
        self.__fitness = 0

        # Add input and output nodes to their respective lists
        for i in range(input_length):
            self.__nodes.append(Node(self.__node_id_index, 'input'))
            self.__node_id_index += 1

        for i in range(output_length):
            self.__nodes.append(Node(self.__node_id_index, 'output'))
            self.__node_id_index += 1

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
            inn_num = self.__inn_num_gen.send((node2, node1))
            conn = Connection(inn_num, node2, node1, weight=weight)
        else:
            inn_num = self.__inn_num_gen.send((node1, node2))
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

        # Create new connections to add in place of the disabled connection
        conn_id_1 = self.__inn_num_gen.send((self.__node_id_index, conn.get_out_node()))
        conn_id_2 = self.__inn_num_gen.send((conn.get_in_node(), self.__node_id_index))
        self.__connections.append(Connection(conn_id_1, self.__node_id_index, conn.get_out_node(), weight=conn.get_weight()))
        self.__connections.append(Connection(conn_id_2, conn.get_in_node(), self.__node_id_index, weight=1.0))

        # Create the node
        self.__nodes.append(Node(self.__node_id_index, 'hidden', activation=activation))
        self.__node_id_index += 1

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

    def get_connection(self, innovation_number):
        """
        Returns the connection with the given innovation number.

        Parameters:
        innovation_number (int): The innovation number of the connection

        Returns:
        (Connection): The connection in the genome with the given innovation number or None if it doesn't exist
        """
        for connection in self.__connections:
            if connection.get_innovation_number() == innovation_number:
                return connection

    def get_connections(self): return self.__connections

    def get_fitness(self): return self.__fitness

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

    def get_nodes(self): return self.__nodes

    def get_node_max_distance(self, node):
        """
        Returns the maximum distance of the given node from an input node.

        Calculated recursively.

        Parameters:
        node (int): The id of the node to test the distance for

        Returns:
        (int): The maximum distance from an input node
        """
        if self.get_node(node).get_type() == 'input':
            return 0
        else:
            distances = []
            for c in self.__connections:
                if c.get_out_node() == node:
                    distances.append(self.get_node_max_distance(c.get_in_node()) + 1)
            return max(distances)

    def mutate_add_connection(self):
        """
        Randomly adds a connection between two existing nodes.
        """
        if self.connections_at_max() is False:
            while True:
                # Select two random, unequal nodes to connect
                node_ids = [node.get_id() for node in self.__nodes]
                node1 = random.choice(node_ids)
                node2 = random.choice(node_ids)

                try:
                    self.add_connection(node1, node2)
                    break
                except GenomeError:
                    continue

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

    def mutate_random_activation(self):
        """
        Randomly selects a hidden node and sets its activation to a random function.
        """
        rand_hidden_node = random.choice([node for node in self.__nodes if node.get_type() == 'hidden'])
        rand_hidden_node.set_activation(activations.get_random())

    def mutate_random_weight(self):
        """
        Randomly selects a connection and sets its weight to a random value.
        """
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

    def set_connections(self, connections): self.__connections = connections

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

    def __eq__(self, conn):
        if isinstance(conn, Connection):
            return conn.get_innovation_number() == self.__innovation_number
        elif isinstance(conn, int):
            return conn == self.__innovation_number
        else:
            return False

    def __str__(self):
        if self.__expressed:
            expressed = 'O'
        else:
            expressed = 'X'
        return '{0:3d}:{1}-{2} [{4}] {3}'.format(self.__innovation_number, self.__in_node, self.__out_node, self.__weight, expressed)

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
