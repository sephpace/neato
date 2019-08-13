
import numpy as np
import random


class NeuralNetwork:
    def __init__(self, input_length, output_length, activation):
        self.input_length = input_length
        self.output_length = output_length
        self.activation = activation
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

    def add_connection(self, id):
        """
        Randomly adds a connection between two existing nodes.
        """
        # TODO: Finish this function so that it actually adds a connections
        # TODO: Find a way to make sure the connection is feed forward
        # TODO: Make sure the connection had the correct id
        rn1 = random.randint(0, len(self.neurons)-1) # The first randomly selected neurons
        rn2 = rn1                                    # The second randomly selected neurons (not equal to rn1)
        while rn2 == rn1:
            rn2 = random.randint(0, len(self.neurons)-1)

    def add_hidden_neuron(self, conn_id_1, conn_id_2):
        """
        Randomly adds a hidden neron on an existing connection.
        """
        # TODO: Find a way to make sure the new connection id's are unique if necessary
        self.neurons.append(Neuron(self.neuron_id_index, 'hidden'))
        rand_conn = self.connections(random.randint(0, len(self.connections)-1))
        rand_conn.enabled = False
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
        self.connections.sort(reverse=True, key=lambda conn: conn.output)


class Neuron:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.value = 0.0


class Connection:
    def __init__(self, id, input, output, weight=None, enabled=True):
        self.id = id
        self.input = input
        self.output = output
        if weight is not None:
            self.weight = weight
        else:
            self.weight = random.random()
        self.enabled = enabled

    def __str__(self):
        enabled = ''
        if self.enabled:
            enabled = 'O'
        else:
            enabled = 'X'
        return '{0}:{1}-{2} {3} [{4}]'.format(self.id, self.input, self.output, self.weight, enabled)


def relu(x): return max(0, x)

# Testing
if __name__ == '__main__':
    nn = NeuralNetwork(3, 2, relu)

    # Manually add nodes and connections for testing
    nn.connections.append(Connection(0, 0, 3))
    nn.connections.append(Connection(1, 1, 3))
    nn.connections.append(Connection(2, 2, 4))
    nn.neurons.append(Neuron(nn.neuron_id_index, 'hidden'))
    nn.neuron_id_index += 1
    nn.connections[0].enabled = False
    nn.connections.append(Connection(3, 5, 3))
    nn.connections.append(Connection(4, 0, 5))
    nn.connections.append(Connection(5, 1, 5))

    print(nn.evaluate([0.5, 0.5, 0.5]))
