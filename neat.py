
import numpy as np
import random


class Network:
    def __init__(self, input_length, output_length, activation, gen_inn_num):
        self.input_length = input_length
        self.output_length = output_length
        self.activation = activation
        self.gen_inn_num = gen_inn_num
        self.neurons = []
        self.connections = []
        self.neuron_id_index = 0

        # Add input and output neurons to their respective lists
        for i in range(input_length):
            self.neurons.append(Neuron(self.neuron_id_index, 'input'))
            self.neuron_id_index += 1

        for i in range(output_length):
            self.neurons.append(Neuron(self.neuron_id_index, 'output'))
            self.neuron_id_index += 1

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
                rn1 = random.randint(0, len(self.neurons)-1) # The first randomly selected neurons
                rn2 = rn1                                    # The second randomly selected neurons (not equal to rn1)
                while rn2 == rn1:
                    rn2 = random.randint(0, len(self.neurons)-1)

                # Get the types of the random nodes
                rn1_type = self.neurons[rn1].type
                rn2_type = self.neurons[rn2].type

                # Don't allow outputs to connect to outputs and inputs to inputs
                if rn1_type == rn2_type == 'input' or rn1_type == rn2_type == 'output':
                    continue

                # Check if the order of the randomly selected neurons should be reversed
                reverse = (rn1_type == 'hidden' and rn2_type == 'input') or (rn1_type == 'output' and rn2_type == 'hidden') or (rn1_type == 'output' and rn2_type == 'input')

                # Create the connection and add it to the network
                if reverse:
                    inn_num = self.gen_inn_num.send((rn2, rn1))
                    conn = Connection(inn_num, rn2, rn1)
                else:
                    inn_num = self.gen_inn_num.send((rn1, rn2))
                    conn = Connection(inn_num, rn1, rn2)

                # Only add the connection if it doesn't already exist within the network
                if conn not in self.connections:
                    self.connections.append(conn)
                    break


    def add_neuron(self):
        """
        Randomly adds a hidden neron on an existing connection.
        """
        self.neurons.append(Neuron(self.neuron_id_index, 'hidden'))
        rand_conn = self.connections[random.randint(0, len(self.connections)-1)]
        rand_conn.enabled = False
        conn_id_1 = self.gen_inn_num.send((self.neuron_id_index, rand_conn.output))
        conn_id_2 = self.gen_inn_num.send((rand_conn.input, self.neuron_id_index))
        self.connections.append(Connection(conn_id_1, self.neuron_id_index, rand_conn.output, weight=rand_conn.weight))
        self.connections.append(Connection(conn_id_2, rand_conn.input, self.neuron_id_index, weight=1.0))
        self.neuron_id_index += 1


    def evaluate(self, inputs):
        """
        Evaluate the neural network and return the outputs.
        """
        # Enter the inputs and reset all non-inputs to zero
        input_index = 0
        for i in range(len(self.neurons)):
            if self.neurons[i].type == 'input':
                self.neurons[i].value = inputs[input_index]
                input_index += 1
            else:
                self.neurons[i].value = 0.0

        # Calculate the outputs
        self.sort_connections()
        for conn in self.connections:
            if conn.enabled:
                input, output = self.neurons[conn.input], self.neurons[conn.output]
                output.value += input.value * conn.weight
                output.value = self.activation(output.value)

        # Return the outputs
        return [neuron.value for neuron in self.neurons if neuron.type == 'output']

    def sort_connections(self):
        """
        Sorts the connections to be in feed-forward order.
        """
        i = 0
        while i < len(self.connections):
            input = self.connections[i].input
            j = i
            while j < len(self.connections) - i:
                if self.connections[j].output == input:
                    self.connections.insert(i, self.connections.pop(j))
                    i -= 2
                    break
                else:
                    j += 1
            i += 1

    def connections_maxxed(self):
        input_amt = self.input_length
        output_amt = self.output_length
        hidden_amt = len(self.neurons) - self.input_length - self.output_length

        maxxed = False
        if hidden_amt == 0:
            maxxed = len(self.connections) == input_amt * output_amt
        else:
            maxxed = len(self.connections) == (input_amt * output_amt) + (input_amt * hidden_amt) + (hidden_amt * output_amt) + (hidden_amt - 1)
        return maxxed


class Neuron:
    """
    A neuron or node in the genome.  Also called a "Node Gene".
    """
    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.value = 0.0


class Connection:
    """
    A connection gene as part of the Genome.
    """
    def __init__(self, id, input, output, weight=None, enabled=True):
        """
        """
        self.id = id
        self.input = input
        self.output = output
        if weight is not None:
            self.weight = weight
        else:
            self.weight = random.random()
        self.enabled = enabled

    def __eq__(self, conn):
        return conn.id == self.id

    def __str__(self):
        enabled = ''
        if self.enabled:
            enabled = 'O'
        else:
            enabled = 'X'
        return '{0}:{1}-{2} [{4}] {3}'.format(self.id, self.input, self.output, self.weight, enabled)


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
    nn = Network(3, 2, relu, gen_inn_num)
    nn.add_connection()
    nn.add_connection()
    nn.add_connection()
    nn.add_hidden_neuron()
    nn.add_connection()
    nn.add_connection()

    print(nn)
    print(nn.evaluate([0.5, 0.5, 0.5]))
