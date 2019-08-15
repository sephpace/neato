
import unittest

from NEAT.genome import Genome, Node, Connection, GenomeError
from NEAT.ecosystem import innovation_number_generator
from activations import relu


class TestGenome(unittest.TestCase):
    def test_constructor(self):
        input_length = 2
        output_length = 2
        activation = relu
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)

        g = Genome(input_length, output_length, activation, inn_num_gen)

        # Test if attributes are correct
        self.assertEqual(g.input_length, input_length)
        self.assertEqual(g.output_length, output_length)
        self.assertEqual(g.activation, activation)
        self.assertEqual(g.inn_num_gen, inn_num_gen)
        self.assertEqual(len(g.nodes), input_length + output_length)
        self.assertEqual(len(g.connections), 0)
        self.assertEqual(g.node_id_index, input_length + output_length)

    def test__add_connection(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, relu, inn_num_gen)

        # Test connecting input and output nodes
        g.add_connection(0, 2)
        self.assertEqual(len(g.connections), 1)
        self.assertEqual(g.connections[0].get_in_node(), 0)
        self.assertEqual(g.connections[0].get_out_node(), 2)

        g.add_connection(0, 3)
        self.assertEqual(len(g.connections), 2)
        self.assertEqual(g.connections[1].get_in_node(), 0)
        self.assertEqual(g.connections[1].get_out_node(), 3)

        g.add_connection(1, 2)
        self.assertEqual(len(g.connections), 3)
        self.assertEqual(g.connections[2].get_in_node(), 1)
        self.assertEqual(g.connections[2].get_out_node(), 2)

        g.add_connection(1, 3, 0.5)
        self.assertEqual(len(g.connections), 4)
        self.assertEqual(g.connections[3].get_in_node(), 1)
        self.assertEqual(g.connections[3].get_out_node(), 3)
        self.assertEqual(g.connections[3].get_weight(), 0.5)

        # Test connecting to hidden nodes
        g.add_node(0)
        g.add_connection(1, 4, 0.7)
        self.assertEqual(len(g.connections), 7)
        self.assertEqual(g.connections[6].get_in_node(), 1)
        self.assertEqual(g.connections[6].get_out_node(), 4)
        self.assertEqual(g.connections[6].get_weight(), 0.7)

        g.add_node(2)
        g.add_connection(4, 5, 0.9)
        self.assertEqual(len(g.connections), 10)
        self.assertEqual(g.connections[9].get_in_node(), 4)
        self.assertEqual(g.connections[9].get_out_node(), 5)
        self.assertEqual(g.connections[9].get_weight(), 0.9)

        # Make sure duplicate genomes aren't created
        g.add_connection(0, 2)
        self.assertEqual(len(g.connections), 10)

        # Make sure you can't connect inputs to inputs or outputs to outputs
        in_to_in = False
        try:
            g.add_connection(0, 1)
        except GenomeError:
            in_to_in = True
        self.assertTrue(in_to_in)

        out_to_out = False
        try:
            g.add_connection(2, 3)
        except GenomeError:
            out_to_out = True
        self.assertTrue(out_to_out)

    def test_add_node(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, relu, inn_num_gen)

        g.add_connection(0, 1)
        g.add_node(0)

        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.connections), 3)
        self.assertFalse(g.connections[0].is_expressed())
        self.assertTrue(g.connections[1].is_expressed())
        self.assertTrue(g.connections[2].is_expressed())
        self.assertEqual(g.connections[0].get_weight(), g.connections[1].get_weight())
        self.assertEqual(g.connections[2].get_weight(), 1.0)
        self.assertEqual(g.node_id_index, 3)


class TestNode(unittest.TestCase):
    pass


class TestConnectoin(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
