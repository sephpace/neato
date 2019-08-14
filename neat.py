
import numpy as np
import random


class Genome:
    def __init__(self, input_length, output_length, activation, gen_inn_num):
        self.input_length = input_length
        self.output_length = output_length
        self.activation = activation
        self.gen_inn_num = gen_inn_num
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
        network_string = '\n---Connections---\n'
        for c in self.connections:
            network_string += str(c) + '\n'
        return network_string

    def add_connection(self):
        """
        Randomly adds a connection between two existing nodes.
        """
        if self.connections_maxxed() is False:
            while True:
                # Select random nodes to connect
                rn1 = random.randint(0, len(self.nodes)-1) # The first randomly selected nodes
                rn2 = rn1                                  # The second randomly selected nodes (not equal to rn1)
                while rn2 == rn1:
                    rn2 = random.randint(0, len(self.nodes)-1)

                # Get the types of the random nodes
                rn1_type = self.nodes[rn1].get_type()
                rn2_type = self.nodes[rn2].get_type()

                # Don't allow outputs to connect to outputs and inputs to inputs
                if rn1_type == rn2_type == 'input' or rn1_type == rn2_type == 'output':
                    continue

                # Check if the order of the randomly selected nodes should be reversed
                reverse = (rn1_type == 'hidden' and rn2_type == 'input') or (rn1_type == 'output' and rn2_type == 'hidden') or (rn1_type == 'output' and rn2_type == 'input')

                # Create the connection and add it to the genome
                if reverse:
                    inn_num = self.gen_inn_num.send((rn2, rn1))
                    conn = Connection(inn_num, rn2, rn1)
                else:
                    inn_num = self.gen_inn_num.send((rn1, rn2))
                    conn = Connection(inn_num, rn1, rn2)

                # Only add the connection if it doesn't already exist within the genome
                if conn not in self.connections:
                    self.connections.append(conn)
                    break


    def add_node(self):
        """
        Randomly adds a hidden node on an existing connection.
        """
        # Create the node
        self.nodes.append(Node(self.node_id_index, 'hidden'))

        # Select a random connection to add the node to and disable it
        rand_conn = self.connections[random.randint(0, len(self.connections)-1)]
        rand_conn.disable()

        # Create new connections to add in place of the randomly selected connection
        conn_id_1 = self.gen_inn_num.send((self.node_id_index, rand_conn.get_out_node()))
        conn_id_2 = self.gen_inn_num.send((rand_conn.get_in_node(), self.node_id_index))
        self.connections.append(Connection(conn_id_1, self.node_id_index, rand_conn.get_out_node(), weight=rand_conn.get_weight()))
        self.connections.append(Connection(conn_id_2, rand_conn.get_in_node(), self.node_id_index, weight=1.0))
        self.node_id_index += 1


    def evaluate(self, inputs):
        """
        Evaluate the neural network (genome) and return the outputs.
        """
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
                input, output = self.nodes[conn.get_in_node()], self.nodes[conn.get_out_node()]
                output.set_value(output.get_value() + input.get_value() * conn.get_weight())
                output.set_value(self.activation(output.get_value()))

        # Return the outputs
        return [node.get_value() for node in self.nodes if node.get_type() == 'output']

    def sort_connections(self):
        """
        Sorts the connections to be in feed-forward order.
        """
        i = 0
        while i < len(self.connections):
            input = self.connections[i].get_in_node()
            j = i
            while j < len(self.connections) - i:
                if self.connections[j].get_out_node() == input:
                    self.connections.insert(i, self.connections.pop(j))
                    i -= 2
                    break
                else:
                    j += 1
            i += 1

    def connections_maxxed(self):
        input_amt = self.input_length
        output_amt = self.output_length
        hidden_amt = len(self.nodes) - self.input_length - self.output_length

        maxxed = False
        if hidden_amt == 0:
            maxxed = len(self.connections) == input_amt * output_amt
        else:
            maxxed = len(self.connections) == (input_amt * output_amt) + (input_amt * hidden_amt) + (hidden_amt * output_amt) + (hidden_amt - 1)
        return maxxed


class Node:
    """
    A node gene in the genome.
    """
    def __init__(self, id, type):
        self.__id = id
        self.__type = type
        self.__value = 0.0

    def get_id(self): return self.__id

    def get_type(self): return self.__type

    def get_value(self): return self.__value

    def set_value(self, value): self.__value = value


class Connection:
    """
    A connection gene as part of the Genome.
    """
    def __init__(self, innovation_number, in_node, out_node, weight=None, expressed=True):
        """
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
        expressed = ''
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


def relu(x): return max(0, x)

def innovation_number_generator():
    inn_log = []
    inn_num = 0
    while True:
        conn = yield inn_num
        if conn in inn_log:
            inn_num = inn_log.index(conn)
        else:
            inn_log.append(conn)
            inn_num = len(inn_log) - 1



# Testing
if __name__ == '__main__':
    gen_inn_num = innovation_number_generator()
    gen_inn_num.send(None)
    g = Genome(3, 2, relu, gen_inn_num)
    g.add_connection()
    g.add_connection()
    g.add_connection()
    g.add_node()
    g.add_connection()
    g.add_connection()

    print(g)
    print(g.evaluate([0.5, 0.5, 0.5]))
