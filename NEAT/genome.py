
import random


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
    __input_length (int):      The amount of input nodes
    __output_length (int):     The amount of output nodes
    __activation (function):   The activation function used by the genome
    __inn_num_gen (generator): The innovation number generator used to assign innovation numbers to connections
    __nodes (list):            The list of nodes in the genome
    __connections (list):      The list of connections in the genome
    __node_id_index (int):     A number that is incremented with each new node to ensure id uniqueness
    """

    def __init__(self, input_length, output_length, activation, inn_num_gen):
        """
        Constructor.

        Parameters:
        input_length (int):      The amount of input nodes
        output_length (int):     The amount of output nodes
        activation (function):   The activation function used by the genome
        inn_num_gen (generator): The innovation number generator used to assign innovation numbers to connections
        """
        self.input_length = input_length
        self.output_length = output_length
        self.activation = activation
        self.inn_num_gen = inn_num_gen
        self.nodes = []
        self.connections = []
        self.node_id_index = 0

        # Add input and output nodes to their respective lists
        for i in range(input_length):
            self.nodes.append(Node(self.node_id_index, 'input'))
            self.node_id_index += 1

        for i in range(output_length):
            self.nodes.append(Node(self.node_id_index, 'output'))
            self.node_id_index += 1

    def __str__(self):
        """
        Represent the genome's node and connection data as a string.

        Returns:
        str: The genome's connection data
        """
        self.nodes.sort(key=lambda n: n.get_id())
        self.connections.sort(key=lambda c: c.get_innovation_number())
        genome_string = '\n-------------Nodes-------------\n'
        for n in self.nodes:
            genome_string += str(n) + '\n'
        genome_string += '\n----------Connections----------\n'
        for c in self.connections:
            genome_string += str(c) + '\n'
        return genome_string

    def add_connection(self, node1, node2, weight=None):
        """
        Adds a connection between node1 and node2.

        Parameters:
        node1 (int): The id of the first node
        node2 (int): The id of the second node
        """
        # Get the types of the random nodes
        node1_type = self.nodes[node1].get_type()
        node2_type = self.nodes[node2].get_type()

        # Don't allow outputs to connect to outputs and inputs to inputs
        if node1_type == node2_type == 'input' or node1_type == node2_type == 'output':
            raise GenomeError('Invalid connection! Cannot connect nodes of type {0} and {1}!'.format(node1_type, node2_type))

        # Check if the order of the randomly selected nodes should be reversed
        reverse = (node1_type == 'hidden' and node2_type == 'input') or (node1_type == 'output' and node2_type == 'hidden') or (node1_type == 'output' and node2_type == 'input')

        # Create the connection and add it to the genome
        if reverse:
            inn_num = self.inn_num_gen.send((node2, node1))
            conn = Connection(inn_num, node2, node1, weight=weight)
        else:
            inn_num = self.inn_num_gen.send((node1, node2))
            conn = Connection(inn_num, node1, node2, weight=weight)

        # Only add the connection if it doesn't already exist within the genome
        if conn not in self.connections:
            self.connections.append(conn)

    def add_node(self, innovation_number):
        """
        Adds a hidden node onto an existing connection with the given innovation number.

        The existing connection is disabled and two new connections are added in its place.

        Parameters:
        innovation_number (int): The innovation number of the connection to add the node to
        """
        # Find the connection to add the node to and disable it
        conn_index = None
        for i in range(len(self.connections)):
            if self.connections[i].get_innovation_number() == innovation_number:
                conn_index = i
                break
        if conn_index is None:
            raise GenomeError('Connection with innovation number {0} does not exist within genome!'.format(innovation_number))
        conn = self.connections[conn_index]
        conn.disable()

        # Create the node
        self.nodes.append(Node(self.node_id_index, 'hidden'))

        # Create new connections to add in place of the disabled connection
        conn_id_1 = self.inn_num_gen.send((self.node_id_index, conn.get_out_node()))
        conn_id_2 = self.inn_num_gen.send((conn.get_in_node(), self.node_id_index))
        self.connections.append(Connection(conn_id_1, self.node_id_index, conn.get_out_node(), weight=conn.get_weight()))
        self.connections.append(Connection(conn_id_2, conn.get_in_node(), self.node_id_index, weight=1.0))
        self.node_id_index += 1

    def connections_at_max(self):
        """
        Returns true if there are no possible places for new connections that are not already filled.

        Returns:
        bool: True if there are no possible places for new connections that are not already filled
        """
        input_amt = self.input_length
        output_amt = self.output_length
        hidden_amt = len(self.nodes) - self.input_length - self.output_length

        if hidden_amt == 0:
            at_max = len(self.connections) == input_amt * output_amt
        else:
            at_max = len(self.connections) == (input_amt * output_amt) + (input_amt * hidden_amt) + (hidden_amt * output_amt) + (hidden_amt - 1)
        return at_max

    def evaluate(self, inputs):
        """
        Evaluate the neural network (genome) and return the outputs.

        Parameters:
        inputs (list): A list of input floats of length equal to the amount of input nodes

        Returns:
        list: A list of output floats between of length equal to the amount of output nodes
        """
        if len(inputs) != self.input_length:
            raise ValueError('Invalid input!  Amount required: {0}  Amount given: {1}'.format(self.input_length, len(inputs)))

        # Enter the inputs and reset all non-inputs to zero
        input_index = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].get_type() == 'input':
                self.nodes[i].set_value(inputs[input_index])
                input_index += 1
            else:
                self.nodes[i].set_value(0.0)

        # Calculate the outputs
        self.sort_connections()
        for conn in self.connections:
            if conn.is_expressed():
                in_node, out_node = self.nodes[conn.get_in_node()], self.nodes[conn.get_out_node()]
                out_node.set_value(out_node.get_value() + in_node.get_value() * conn.get_weight())
                out_node.set_value(self.activation(out_node.get_value()))

        # Return the outputs
        return [node.get_value() for node in self.nodes if node.get_type() == 'output']

    def mutate_add_connection(self):
        """
        Randomly adds a connection between two existing nodes.
        """
        if self.connections_at_max() is False:
            while True:
                # Select two random, unequal nodes to connect
                node1 = random.randint(0, len(self.nodes)-1)
                node2 = node1
                while node2 == node1:
                    node2 = random.randint(0, len(self.nodes)-1)

                # Get the types of the random nodes
                node1_type = self.nodes[node1].get_type()
                node2_type = self.nodes[node2].get_type()

                # Don't allow outputs to connect to outputs and inputs to inputs
                if node1_type == node2_type == 'input' or node1_type == node2_type == 'output':
                    continue

                # Check if the order of the randomly selected nodes should be reversed
                reverse = (node1_type == 'hidden' and node2_type == 'input') or (node1_type == 'output' and node2_type == 'hidden') or (node1_type == 'output' and node2_type == 'input')

                # Create the connection and add it to the genome
                if reverse:
                    inn_num = self.inn_num_gen.send((node2, node1))
                    conn = Connection(inn_num, node2, node1)
                else:
                    inn_num = self.inn_num_gen.send((node1, node2))
                    conn = Connection(inn_num, node1, node2)

                # Only add the connection if it doesn't already exist within the genome
                if conn not in self.connections:
                    self.connections.append(conn)
                    break

    def mutate_add_node(self):
        """
        Randomly adds a hidden node onto an existing connection.

        The existing connection is disabled and two new connections are added in its place.
        """
        # Create the node
        self.nodes.append(Node(self.node_id_index, 'hidden'))

        # Select a random connection to add the node to and disable it
        rand_conn = self.connections[random.randint(0, len(self.connections)-1)]
        rand_conn.disable()

        # Create new connections to add in place of the randomly selected connection
        conn_id_1 = self.inn_num_gen.send((self.node_id_index, rand_conn.get_out_node()))
        conn_id_2 = self.inn_num_gen.send((rand_conn.get_in_node(), self.node_id_index))
        self.connections.append(Connection(conn_id_1, self.node_id_index, rand_conn.get_out_node(), weight=rand_conn.get_weight()))
        self.connections.append(Connection(conn_id_2, rand_conn.get_in_node(), self.node_id_index, weight=1.0))
        self.node_id_index += 1

    def sort_connections(self):
        """
        Sorts the connections to be in feed-forward order.
        """
        i = 0
        while i < len(self.connections):
            in_node = self.connections[i].get_in_node()
            j = i
            while j < len(self.connections) - i:
                if self.connections[j].get_out_node() == in_node:
                    self.connections.insert(i, self.connections.pop(j))
                    i -= 2
                    break
                else:
                    j += 1
            i += 1


class Node:
    """
    A node gene in the genome.

    Attributes:
    __id_num (int):    The node's id number within the genome
    __node_type (str): The type of node ('input', 'output', or 'hidden')
    __value (float):   The current value contained in the node (the input for input nodes, the output for output nodes, or the placeholder value for hidden nodes)
    """
    def __init__(self, id_num, node_type):
        """
        Constructor.

        Parameters:
        id (int):        The node's id number within the genome
        node_type (str): The type of node ('input', 'output', or 'hidden')
        """
        self.__id_num = id_num
        self.__node_type = node_type
        self.__value = 0.0

    def __str__(self):
        return '{0}: {1:6s} {2}'.format(self.__id_num, self.__node_type, self.__value)

    def get_id(self): return self.__id_num

    def get_type(self): return self.__node_type

    def get_value(self): return self.__value

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
            self.__weight = random.random()
        self.__expressed = expressed

    def __eq__(self, conn):
        return conn.get_innovation_number() == self.__innovation_number

    def __str__(self):
        if self.__expressed:
            expressed = 'O'
        else:
            expressed = 'X'
        return '{0}:{1}-{2} [{4}] {3}'.format(self.__innovation_number, self.__in_node, self.__out_node, self.__weight, expressed)

    def enable(self): self.__expressed = True

    def disable(self): self.__expressed = False

    def is_expressed(self): return self.__expressed

    def get_innovation_number(self): return self.__innovation_number

    def get_in_node(self): return self.__in_node

    def get_out_node(self): return self.__out_node

    def get_weight(self): return self.__weight

    def set_weight(self, weight): self.__weight = weight


class GenomeError(Exception):
    def __init__(self, message):
        self.message = message
