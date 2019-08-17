
import unittest

from genome import Genome, Node, Connection, GenomeError
from ecosystem import innovation_number_generator
import activations


class TestEcosystem(unittest.TestCase):
    pass


class Test_Innovation_Number_Generator(unittest.TestCase):
    def test_standalone(self):
        msg = 'Innovation number incorrectly generated!'

        # Test to make sure innovation numbers are always unique
        inn = innovation_number_generator()
        inn.send(None)
        count = 0
        for i in range(33):
            for j in range(33):
                self.assertEqual(inn.send((i, j)), count, msg)
                count += 1

        self.assertEqual(inn.send((32, 33)), 1089, msg)

        # It should get the same results twice
        count = 0
        for i in range(33):
            for j in range(33):
                self.assertEqual(inn.send((i, j)), count, msg)
                count += 1

        self.assertEqual(inn.send((32, 33)), 1089, msg)

    def test_one_genome(self):
        msg = 'Innovation number incorrectly generated!'

        # Test to make sure innovation numbers are always unique
        inn = innovation_number_generator()
        inn.send(None)
        g = Genome(5, 5, inn)

        for i in range(25):
            g.mutate_add_connection()

        for i in range(25):
            self.assertTrue(i in [c.get_innovation_number() for c in g.connections], msg)

    def test_two_genomes(self):
        msg = 'Innovation number incorrectly generated!'

        # Test to make sure innovation numbers are always unique
        inn = innovation_number_generator()
        inn.send(None)

        g1 = Genome(5, 5, inn)
        g2 = Genome(5, 5, inn)

        for i in range(50):
            if i % 2 == 0: # Even numbers
                g1.mutate_add_connection()
            else:          # Odd numbers
                g2.mutate_add_connection()

        for i in range(25):
            self.assertTrue(i in [c.get_innovation_number() for c in g1.connections], msg)
            self.assertTrue(i in [c.get_innovation_number() for c in g2.connections], msg)



class TestGenome(unittest.TestCase):

    def test_constructor(self):
        input_length = 2
        output_length = 2
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)

        g = Genome(input_length, output_length, inn_num_gen)

        # Test if attributes are correct
        msg = 'Failed to assign genome attributes correctly!'
        self.assertEqual(g.input_length, input_length, msg)
        self.assertEqual(g.output_length, output_length, msg)
        self.assertEqual(g.inn_num_gen, inn_num_gen, msg)
        self.assertEqual(len(g.nodes), input_length + output_length, msg)
        self.assertEqual(len(g.connections), 0)
        self.assertEqual(g.node_id_index, input_length + output_length, msg)

    def test__add_connection(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        # Test connecting input and output nodes
        msg = 'Failed connection to output node!'
        g.add_connection(0, 2)
        self.assertEqual(len(g.connections), 1, msg)
        self.assertEqual(g.connections[0].get_in_node(), 0, msg)
        self.assertEqual(g.connections[0].get_out_node(), 2, msg)

        g.add_connection(0, 3)
        self.assertEqual(len(g.connections), 2, msg)
        self.assertEqual(g.connections[1].get_in_node(), 0, msg)
        self.assertEqual(g.connections[1].get_out_node(), 3, msg)

        g.add_connection(1, 2)
        self.assertEqual(len(g.connections), 3, msg)
        self.assertEqual(g.connections[2].get_in_node(), 1, msg)
        self.assertEqual(g.connections[2].get_out_node(), 2, msg)

        g.add_connection(1, 3, 0.5)
        self.assertEqual(len(g.connections), 4, msg)
        self.assertEqual(g.connections[3].get_in_node(), 1, msg)
        self.assertEqual(g.connections[3].get_out_node(), 3, msg)
        self.assertEqual(g.connections[3].get_weight(), 0.5, msg)

        # Test connecting to hidden nodes
        msg = 'Failed connection to hidden node!'
        g.add_node(0)
        g.add_connection(1, 4, 0.7)
        self.assertEqual(len(g.connections), 7, msg)
        self.assertEqual(g.connections[6].get_in_node(), 1, msg)
        self.assertEqual(g.connections[6].get_out_node(), 4, msg)
        self.assertEqual(g.connections[6].get_weight(), 0.7, msg)

        g.add_node(2)
        g.add_connection(4, 5, 0.9)
        self.assertEqual(len(g.connections), 10, msg)
        self.assertEqual(g.connections[9].get_in_node(), 4, msg)
        self.assertEqual(g.connections[9].get_out_node(), 5, msg)
        self.assertEqual(g.connections[9].get_weight(), 0.9, msg)

        # Make sure duplicate connections aren't created
        msg = 'Duplicate connection added!'
        try:
            g.add_connection(0, 2)
        except GenomeError:
            pass
        self.assertEqual(len(g.connections), 10, msg)

        # Make sure backwards connections aren't created
        msg = 'Backward connection added!'
        try:
            g.add_connection(5, 4)
        except GenomeError:
            pass
        self.assertEqual(len(g.connections), 10, msg)

        # Make sure you can't connect inputs to inputs or outputs to outputs
        msg = 'Input node connected to input node!'
        in_to_in = False
        try:
            g.add_connection(0, 1)
        except GenomeError:
            in_to_in = True
        self.assertTrue(in_to_in, msg)

        msg = 'Output node connected to output node!'
        out_to_out = False
        try:
            g.add_connection(2, 3)
        except GenomeError:
            out_to_out = True
        self.assertTrue(out_to_out, msg)

        # Make sure you can't connect a node to itself
        msg = 'Node connected to itself!'
        node_to_itself = False
        try:
            g.add_connection(6, 6)
        except GenomeError:
            node_to_itself = True
        self.assertTrue(node_to_itself, msg)

    def test_add_node(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, inn_num_gen)

        # Test adding nodes
        msg = 'Node added incorrectly!'

        g.add_connection(0, 1)
        g.add_node(0)
        self.assertEqual(len(g.nodes), 3, msg)
        self.assertEqual(len(g.connections), 3, msg)
        self.assertFalse(g.connections[0].is_expressed(), msg)
        self.assertTrue(g.connections[1].is_expressed(), msg)
        self.assertTrue(g.connections[2].is_expressed(), msg)
        self.assertEqual(g.connections[0].get_weight(), g.connections[1].get_weight(), msg)
        self.assertEqual(g.connections[2].get_weight(), 1.0, msg)
        self.assertEqual(g.node_id_index, 3, msg)
        self.assertEqual(g.nodes[2].get_activation(), activations.modified_sigmoid, msg)

        g.add_node(1, activation=activations.absolute)
        self.assertEqual(len(g.nodes), 4, msg)
        self.assertEqual(len(g.connections), 5, msg)
        self.assertFalse(g.connections[1].is_expressed(), msg)
        self.assertTrue(g.connections[3].is_expressed(), msg)
        self.assertTrue(g.connections[4].is_expressed(), msg)
        self.assertEqual(g.connections[1].get_weight(), g.connections[3].get_weight(), msg)
        self.assertEqual(g.connections[4].get_weight(), 1.0, msg)
        self.assertEqual(g.node_id_index, 4, msg)
        self.assertEqual(g.nodes[3].get_activation(), activations.absolute, msg)

    def test_connections_at_max(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        msg = 'Connections not at max and should be!'
        msg2 = 'Connections at max and shouldn\'t be!'

        # Test with no hidden nodes
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(1, 3)
        self.assertTrue(g.connections_at_max(), msg2)

        # Test with hidden nodes
        g.add_node(0)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(1, 4)
        g.add_connection(4, 3)
        self.assertTrue(g.connections_at_max(), msg2)
        g.add_node(6)
        self.assertFalse(g.connections_at_max(), msg)
        g.add_connection(0, 5)
        g.add_connection(5, 2)
        g.add_connection(5, 3)
        self.assertTrue(g.connections_at_max(), msg2)

    def test_evaluate(self):
        error_margin = 0.000000000001

        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        msg = 'Invalid evaluation result!'

        # No hidden nodes, modified sigmoid activation
        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        results = g.evaluate([0.5, 0.5])
        self.assertAlmostEqual(results[0], 0.3798935676569099, msg=msg, delta=error_margin)
        self.assertAlmostEqual(results[1], 0.8765329524347759, msg=msg, delta=error_margin)

        # Different activation
        for n in g.nodes:
            n.set_activation(activations.absolute)
        results = g.evaluate([0.5, 0.5])
        self.assertAlmostEqual(results[0], 0.1, msg=msg, delta=error_margin)
        self.assertAlmostEqual(results[1], 0.4, msg=msg, delta=error_margin)

        # With hidden nodes and different activations (sigmoid for the new ones)
        g.add_node(0, activation=activations.sigmoid)
        g.add_node(2, activation=activations.sigmoid)
        g.add_connection(4, 5, 0.5)
        results = g.evaluate([0.5, 0.5])
        self.assertAlmostEqual(results[0], 0.08953579350695234, msg=msg, delta=error_margin)
        self.assertAlmostEqual(results[1], 0.4, msg=msg, delta=error_margin)

    def test_get_node_max_distance(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        g.add_node(0)
        g.add_node(2)
        g.add_connection(4, 5, 0.5)

        msg = 'Incorrect node max distance!'

        # Test the values of each node distance to make sure they are correct
        correct_distances = [0, 0, 3, 1, 1, 2]
        for i in range(len(g.nodes)):
            self.assertEqual(g.get_node_max_distance(g.nodes[i].get_id()), correct_distances[i], msg)

        # Add a node and test again
        g.add_node(8)
        correct_distances = [0, 0, 4, 1, 1, 3, 2]
        for i in range(len(g.nodes)):
            self.assertEqual(g.get_node_max_distance(g.nodes[i].get_id()), correct_distances[i], msg)

        # Add connection and test again
        g.add_connection(6, 3)
        correct_distances = [0, 0, 4, 3, 1, 3, 2]
        for i in range(len(g.nodes)):
            self.assertEqual(g.get_node_max_distance(g.nodes[i].get_id()), correct_distances[i], msg)

    def test_mutate_add_connection(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        # Test adding a connection
        msg = 'Connection added incorrectly!'
        g.mutate_add_connection()
        self.assertEqual(len(g.connections), 1, msg)
        self.assertEqual(g.connections[0].get_innovation_number(), 0, msg)
        self.assertTrue(g.connections[0].get_in_node() in [n.get_id() for n in g.nodes], msg)
        self.assertTrue(g.connections[0].get_out_node() in [n.get_id() for n in g.nodes], msg)
        self.assertTrue(-1.0 <= g.connections[0].get_weight() <= 1.0, msg)
        self.assertTrue(g.connections[0].is_expressed(), msg)
        self.assertNotEqual(g.connections[0].get_in_node(), g.connections[0].get_out_node())

        # Test to make sure connections are always added (unless at max)
        msg = 'Connection not added!'
        g.mutate_add_connection()
        self.assertEqual(len(g.connections), 2, msg)
        g.mutate_add_connection()
        self.assertEqual(len(g.connections), 3, msg)
        g.mutate_add_connection()
        self.assertEqual(len(g.connections), 4, msg)

        # Test to make sure it doesn't go above the maximum connections
        msg = 'Connections exceeded maximum amount!'
        g.mutate_add_connection()
        self.assertEqual(len(g.connections), 4, msg)  # Shouldn't go passed 4

    def test_mutate_add_node(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, inn_num_gen)

        # Test to make sure you can't add a node without an existing connection
        msg = 'Node added without existing connection!'
        g.mutate_add_node()
        self.assertEqual(len(g.nodes), 2, msg)

        # Test adding a node
        msg = 'Node added incorrectly!'
        g.mutate_add_connection()
        g.mutate_add_node()
        self.assertEqual(len(g.nodes), 3, msg)
        self.assertEqual(g.nodes[2].get_id(), 2, msg)
        self.assertEqual(g.nodes[2].get_type(), 'hidden', msg)
        self.assertEqual(g.nodes[2].get_value(), 0.0, msg)
        self.assertEqual(len(g.connections), 3, msg)
        self.assertEqual(g.connections[0].get_weight(), g.connections[1].get_weight(), msg)
        self.assertEqual(g.connections[2].get_weight(), 1.0, msg)

        # Test to make sure you can't add a node without any expressed connections
        msg = "Node added to disabled connection!"
        for c in g.connections:
            c.disable()
        g.mutate_add_node()
        self.assertEqual(len(g.nodes), 3, msg)

    def test_mutate_random_weight(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, inn_num_gen)

        msg = 'Failed to set random connection weight!'

        # Test with one connection
        g.add_connection(0, 1, weight=5)
        g.mutate_random_weight()
        self.assertNotEqual(g.connections[0].get_weight(), 5, msg)
        self.assertTrue(-1.0, g.connections[0].get_weight() <= 1.0)

        # Test with multiple connections
        g.add_node(0)
        before = [c.get_weight() for c in g.connections]
        g.mutate_random_weight()
        after = [c.get_weight() for c in g.connections]
        self.assertNotEqual(before, after, msg)

    def test_mutate_shift_weight(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, inn_num_gen)

        msg = 'Failed to shift weight correctly!'

        # Test with one connection
        g.add_connection(0, 1, weight=0.0)
        g.mutate_shift_weight(step=0.1)
        self.assertNotEqual(g.connections[0].get_weight(), 0.0, msg)
        self.assertTrue(g.connections[0].get_weight() == 0.1 or g.connections[0].get_weight() == -0.1)

        # Test with multiple connections
        g.add_node(0)
        before = [c.get_weight() for c in g.connections]
        g.mutate_shift_weight()
        after = [c.get_weight() for c in g.connections]
        self.assertNotEqual(before, after, msg)

    def test_mutate_toggle_connection(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(1, 1, inn_num_gen)

        msg = 'Failed to toggle connection!'

        # Test with one connection
        g.add_connection(0, 1)
        g.mutate_toggle_connection()
        self.assertFalse(g.connections[0].is_expressed(), msg)
        g.mutate_toggle_connection()
        self.assertTrue(g.connections[0].is_expressed(), msg)

        # Test with multiple connections
        g.add_node(0)
        before = [c.is_expressed() for c in g.connections]
        g.mutate_toggle_connection()
        after = [c.is_expressed() for c in g.connections]
        self.assertNotEqual(before, after, msg)

    def test_sort_connections(self):
        inn_num_gen = innovation_number_generator()
        inn_num_gen.send(None)
        g = Genome(2, 2, inn_num_gen)

        # Create a fairly complex structure
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)
        g.add_connection(1, 3)
        g.add_node(3)
        g.add_node(5)
        g.add_node(6)
        g.add_node(7)
        g.add_connection(4, 7)
        g.add_connection(1, 6)
        g.add_connection(0, 6)
        g.add_connection(7, 3)
        g.add_node(0)
        g.add_node(2)
        g.add_connection(8, 7)
        g.add_connection(4, 9)

        # Test to make sure all of the connections are sorted in a feed-forward manner
        msg = 'Connections are not feed-forward!'
        g.sort_connections()
        for i in range(len(g.connections)):
            self.assertTrue(g.connections[i].get_in_node() not in [c.get_out_node() for c in g.connections[i:]], msg)


class TestNode(unittest.TestCase):
    def test_constructor(self):
        # Test node attribute values
        msg = 'Failed to assign node attributes correctly!'
        n = Node(0, 'input')
        self.assertEqual(n.get_id(), 0, msg)
        self.assertEqual(n.get_type(), 'input', msg)
        self.assertIsNone(n.get_activation(), msg)
        self.assertEqual(n.get_value(), 0.0, msg)

        n2 = Node(1, 'output', activation=activations.sigmoid, value=5.4)
        self.assertEqual(n2.get_id(), 1, msg)
        self.assertEqual(n2.get_type(), 'output', msg)
        self.assertEqual(n2.get_activation(), activations.sigmoid, msg)
        self.assertEqual(n2.get_value(), 5.4, msg)

        n3 = Node(2, 'hidden', activation=activations.absolute, value=-0.3)
        self.assertEqual(n3.get_id(), 2, msg)
        self.assertEqual(n3.get_type(), 'hidden', msg)
        self.assertEqual(n3.get_activation(), activations.absolute, msg)
        self.assertEqual(n3.get_value(), -0.3, msg)

    def test_activate(self):
        # Test to make sure the value is run through the activation function correctly
        msg = 'Failed to set node activation correctly!'
        n = Node(0, 'hidden', value=-0.8)
        n.activate()
        self.assertAlmostEqual(n.get_value(), 0.01945508456819303, msg=msg, delta=0.000000000001)

        n2 = Node(0, 'hidden', activation=activations.absolute, value=-1.0)
        n2.activate()
        self.assertEqual(n2.get_value(), 1.0, msg)

    def test_set_activation(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to set node activation correctly!'
        n = Node(0, 'hidden')
        n.set_activation(activations.absolute)
        self.assertEqual(n.get_activation(), activations.absolute, msg)

        # Test to make sure input nodes cannot have an activation
        msg = 'Input nodes cannot have activation functions!'
        n = Node(1, 'input')
        n.set_activation(activations.absolute)
        self.assertIsNone(n.get_activation(), msg)

    def test_set_value(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to set node value correctly!'
        n = Node(0, 'input')
        n.set_value(39)
        self.assertEqual(n.get_value(), 39, msg)


class TestConnection(unittest.TestCase):
    def test_constructor(self):
        # Test connection attribute values
        msg = 'Failed to assign connection attributes correctly!'
        c = Connection(0, 0, 2)
        self.assertEqual(c.get_innovation_number(), 0, msg)
        self.assertEqual(c.get_in_node(), 0, msg)
        self.assertEqual(c.get_out_node(), 2, msg)
        self.assertTrue(-1.0 <= c.get_weight() <= 1.0, msg)
        self.assertTrue(c.is_expressed(), msg)

        c2 = Connection(1, 57, 9, weight=-0.4, expressed=False)
        self.assertEqual(c2.get_innovation_number(), 1, msg)
        self.assertEqual(c2.get_in_node(), 57, msg)
        self.assertEqual(c2.get_out_node(), 9, msg)
        self.assertEqual(c2.get_weight(), -0.4, msg)
        self.assertFalse(c2.is_expressed(), msg)

    def test_disable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to disable connection correctly!'
        c = Connection(0, 0, 2)
        c.disable()
        self.assertFalse(c.is_expressed(), msg)

    def test_enable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to enable connection correctly'
        c = Connection(0, 0, 2, expressed=False)
        c.enable()
        self.assertTrue(c.is_expressed(), msg)

    def test_set_weight(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2)
        c.set_weight(117)
        self.assertEqual(c.get_weight(), 117, msg)

    def test_set_random_weight(self):
        # Test to make sure it returns a random value between -1 and 1
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2, weight=5)
        c.set_random_weight()
        self.assertNotEqual(c.get_weight(), 5, msg)
        self.assertTrue(-1.0 <= c.get_weight() <= 1.0, msg)

    def test_toggle(self):
        # Test to make sure the connection's expression toggles correctly
        msg = 'Failed to toggle connection\'s expression correctly!'
        c = Connection(0, 0, 2)
        c.toggle()
        self.assertFalse(c.is_expressed(), msg)
        c.toggle()
        self.assertTrue(c.is_expressed(), msg)

if __name__ == '__main__':
    unittest.main()
