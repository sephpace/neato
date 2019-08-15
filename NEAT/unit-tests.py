
import unittest

from genome import Genome, Node, Connection, GenomeError
from ecosystem import innovation_number_generator


class TestGenome(unittest.TestCase):
    relu = lambda self, x: max(0, x)

    def test_constructor(self):
        input_length = 2
        output_length = 2
        activation = self.relu
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
        g = Genome(2, 2, self.relu, inn_num_gen)

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
        g = Genome(1, 1, self.relu, inn_num_gen)

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

        g.add_node(1)

        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(len(g.connections), 5)
        self.assertFalse(g.connections[1].is_expressed())
        self.assertTrue(g.connections[3].is_expressed())
        self.assertTrue(g.connections[4].is_expressed())
        self.assertEqual(g.connections[1].get_weight(), g.connections[3].get_weight())
        self.assertEqual(g.connections[4].get_weight(), 1.0)
        self.assertEqual(g.node_id_index, 4)

    def test_connections_at_max(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, self.relu, inn_num_gen)

        # Test with no hidden nodes
        self.assertFalse(g.connections_at_max())
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)
        self.assertFalse(g.connections_at_max())
        g.add_connection(1, 3)
        self.assertTrue(g.connections_at_max())

        # Test with hidden nodes
        g.add_node(0)
        self.assertFalse(g.connections_at_max())
        g.add_connection(1, 4)
        g.add_connection(4, 3)
        self.assertTrue(g.connections_at_max())
        g.add_node(6)
        self.assertFalse(g.connections_at_max())
        g.add_connection(0, 5)
        g.add_connection(5, 2)
        g.add_connection(5, 3)
        self.assertTrue(g.connections_at_max())

    def test_evalutate(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, self.relu, inn_num_gen)

        # No hidden nodes, relu activation
        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        results = g.evaluate([0.5, 0.5])
        self.assertEqual(results[0], 0.0)
        self.assertEqual(results[1], 0.4)

        # Different activation
        g.activation = lambda self, x: x+1
        results = g.evaluate([0.5, 0.5])
        self.assertEqual(results[0], 0.9)
        self.assertEqual(results[1], 1.4)

        # With hidden nodes
        g.add_node(0)
        g.add_node(2)
        g.add_connection(4, 5, 0.5)
        results = g.evaluate([0.5, 0.5])
        self.assertEqual(results[0], 1.0750000000000002)
        self.assertEqual(results[1], 1.4)






class TestNode(unittest.TestCase):
    pass


class TestConnectoin(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
