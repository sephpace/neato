
import unittest

import numpy as np

from neato import activations
from neato.genome import Genome, Node, Connection


class TestGenome(unittest.TestCase):

    def test_constructor(self):
        input_size = 2
        output_size = 2

        g = Genome(input_size, output_size)

        # Test if attributes are correct
        msg = 'Failed to assign genome attributes correctly!'
        self.assertEqual(len(g.get_nodes()), input_size + output_size, msg)
        self.assertEqual(len(g.get_connections()), 0)
        self.assertEqual(g.fitness, 0)

    def test__add_connection(self):
        g = Genome(2, 2)

        # Test connecting input and output nodes
        msg = 'Failed connection to output node!'
        g.add_connection(0, 2)
        self.assertEqual(len(g.get_connections()), 1, msg)
        self.assertEqual(g.get_connections()[0].in_node, 0, msg)
        self.assertEqual(g.get_connections()[0].out_node, 2, msg)

        g.add_connection(0, 3)
        self.assertEqual(len(g.get_connections()), 2, msg)
        self.assertEqual(g.get_connections()[1].in_node, 0, msg)
        self.assertEqual(g.get_connections()[1].out_node, 3, msg)

        g.add_connection(1, 2)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertEqual(g.get_connections()[2].in_node, 1, msg)
        self.assertEqual(g.get_connections()[2].out_node, 2, msg)

        g.add_connection(1, 3, 0.5)
        self.assertEqual(len(g.get_connections()), 4, msg)
        self.assertEqual(g.get_connections()[3].in_node, 1, msg)
        self.assertEqual(g.get_connections()[3].out_node, 3, msg)
        self.assertEqual(g.get_connections()[3].weight, 0.5, msg)

        # # Test connecting to hidden nodes
        msg = 'Failed connection to hidden node!'
        g.add_node(0)
        g.add_connection(1, 4, 0.7)
        self.assertEqual(len(g.get_connections()), 7, msg)
        self.assertEqual(g.get_connections()[6].in_node, 1, msg)
        self.assertEqual(g.get_connections()[6].out_node, 4, msg)
        self.assertEqual(g.get_connections()[6].weight, 0.7, msg)

        g.add_node(2)
        g.add_connection(4, 5, 0.9)
        self.assertEqual(len(g.get_connections()), 10, msg)
        self.assertEqual(g.get_connections()[9].in_node, 4, msg)
        self.assertEqual(g.get_connections()[9].out_node, 5, msg)
        self.assertEqual(g.get_connections()[9].weight, 0.9, msg)

        # Make sure duplicate connections aren't created
        msg = 'Duplicate connection added!'
        try:
            g.add_connection(0, 2)
        except AssertionError:
            pass
        self.assertEqual(len(g.get_connections()), 10, msg)

    def test_add_node(self):
        g = Genome(1, 1)

        # Test adding nodes
        msg = 'Node added incorrectly!'

        g.add_connection(0, 1)
        g.add_node(0)
        self.assertEqual(len(g.get_nodes()), 3, msg)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertFalse(g.get_connections()[0].expressed, msg)
        self.assertTrue(g.get_connections()[1].expressed, msg)
        self.assertTrue(g.get_connections()[2].expressed, msg)
        self.assertEqual(g.get_connections()[0].weight, g.get_connections()[1].weight, msg)
        self.assertEqual(g.get_connections()[2].weight, 1.0, msg)
        self.assertEqual(g.get_nodes()[2].activation, activations.modified_sigmoid, msg)

        g.add_node(1, activation=activations.absolute)
        self.assertEqual(len(g.get_nodes()), 4, msg)
        self.assertEqual(len(g.get_connections()), 5, msg)
        self.assertFalse(g.get_connections()[1].expressed, msg)
        self.assertTrue(g.get_connections()[3].expressed, msg)
        self.assertTrue(g.get_connections()[4].expressed, msg)
        self.assertEqual(g.get_connections()[1].weight, g.get_connections()[3].weight, msg)
        self.assertEqual(g.get_connections()[4].weight, 1.0, msg)
        self.assertEqual(g.get_nodes()[3].activation, activations.absolute, msg)

    def test_connections_at_max(self):
        g = Genome(2, 2)

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

    def test_copy(self):
        g = Genome(2, 2)

        # Test to make sure the copy is the same as the original
        msg = 'Copy is different from the original!'
        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_node(0)
        gc = g.copy()
        self.assertEqual(len(g.get_nodes()), len(gc.get_nodes()), msg)
        self.assertEqual(len(g.get_connections()), len(gc.get_connections()), msg)
        for i in range(len(g.get_connections())):
            self.assertEqual(g.get_connections()[i].weight, gc.get_connections()[i].weight, msg)

        # Test to make sure the copy doesn't change when the original does
        msg = 'Copy changes when original changes!'
        g.add_node(1)
        g.get_connections()[0].weight = 50
        self.assertNotEqual(len(g.get_nodes()), len(gc.get_nodes()), msg)
        self.assertNotEqual(len(g.get_connections()), len(gc.get_connections()), msg)
        self.assertNotEqual(g.get_connections()[0].weight, gc.get_connections()[0].weight, msg)

    def test_forward(self):
        error_margin = 0.000000000001

        g = Genome(2, 2)
        x = np.array([0.5, 0.5])

        msg = 'Invalid genome output!'

        # No hidden nodes, modified sigmoid activation
        g.add_connection(0, 2, weight=-0.7)
        g.add_connection(0, 3, weight=-0.1)
        g.add_connection(1, 2, weight=0.5)
        g.add_connection(1, 3, weight=0.9)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.3798935676569099, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.8765329524347759, msg=msg, delta=error_margin)

        # Different activation
        for n in g.get_nodes():
            if n.type != 'input':
                g.set_activation(n.id, activations.absolute)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.1, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.4, msg=msg, delta=error_margin)

        # With hidden nodes and different activations (sigmoid for the new ones)
        g.add_node(0, activation=activations.sigmoid)
        g.add_node(2, activation=activations.sigmoid)
        g.add_connection(4, 5, 0.5)
        y = g(x)
        self.assertAlmostEqual(y[0], 0.08953579350695234, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.4, msg=msg, delta=error_margin)

        # Test with many hidden nodes in a line
        g2 = Genome(2, 2)
        g2.add_connection(0, 2, -0.3)
        g2.add_node(0)
        g2.add_node(2)
        g2.add_node(4)
        g2.add_node(6)
        g2.add_connection(1, 7, 0.7)
        g2.add_connection(4, 3, -0.2)
        y = g2(x)
        self.assertAlmostEqual(y[0], 0.18866305913528142, msg=msg, delta=error_margin)
        self.assertAlmostEqual(y[1], 0.2743863603871294, msg=msg, delta=error_margin)

        # Test with recursive loop
        g3 = Genome(1, 1)
        x = np.array([0.5])
        g3.add_connection(0, 1, 0.5)
        g3.add_node(0)
        g3.add_node(0)
        g3.add_node(1)
        g3.add_connection(3, 2, 0.5)
        g3.add_connection(4, 3, 0.5)
        y = g3(x)
        self.assertAlmostEqual(y[0], 0.9907948306148218, msg=msg, delta=error_margin)

    def test_get_node_inputs(self):
        g = Genome(2, 2)

        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)

        # Test to make sure it gets the inputs correctly
        msg = 'Node inputs are incorrect!'
        self.assertEqual(g.get_node_inputs(2), [0, 1], msg)
        self.assertEqual(g.get_node_inputs(3), [0], msg)

    def test_get_node_max_distance(self):
        g = Genome(2, 2)

        # Test to make sure empty genomes return the correct values for output nodes
        msg = 'Disconnected output node returned invalid value!'
        self.assertEqual(g.get_node_max_distance(g.get_node(2).id), -1, msg)
        self.assertEqual(g.get_node_max_distance(g.get_node(3).id), -1, msg)

        # Add nodes and connections
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
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Add a node and test again
        g.add_node(8)
        correct_distances = [0, 0, 4, 1, 1, 3, 2]
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Add connection and test again
        g.add_connection(6, 3)
        correct_distances = [0, 0, 4, 3, 1, 3, 2]
        for node, distance in zip(g.get_nodes(), correct_distances):
            self.assertEqual(g.get_node_max_distance(node.id), distance, msg)

        # Test genome with connection loop
        msg = 'Genome failed to handle connection loop!'

        g2 = Genome(1, 1)
        g2.add_connection(0, 1)
        g2.add_node(0)
        g2.add_node(0)
        g2.add_node(1)
        g2.add_connection(3, 2)
        g2.add_connection(4, 3)
        correct_distances = [0, 4, 3, 3, 3]
        for node, distance in zip(g2.get_nodes(), correct_distances):
            self.assertEqual(g2.get_node_max_distance(node.id), distance, msg)

    def test_get_node_outputs(self):
        g = Genome(2, 2)

        g.add_connection(0, 2)
        g.add_connection(0, 3)
        g.add_connection(1, 2)

        # Test to make sure it gets the outputs correctly
        msg = 'Node outputs are incorrect!'
        self.assertEqual(g.get_node_outputs(0), [2, 3], msg)
        self.assertEqual(g.get_node_outputs(1), [2], msg)

    def test_mutate_add_connection(self):
        # Test adding a connection
        msg = 'Connection added incorrectly!'
        g = Genome(2, 2)

        for i in range(30):
            g = Genome(2, 2)
            g.mutate_add_connection()
            self.assertEqual(len(g.get_connections()), 1, msg)
            self.assertEqual(g.get_connections()[0].innovation_number, 0, msg)
            self.assertTrue(g.get_connections()[0].in_node in [n.id for n in g.get_nodes()], msg)
            self.assertTrue(g.get_connections()[0].out_node in [n.id for n in g.get_nodes()], msg)
            self.assertTrue(-1.0 <= g.get_connections()[0].weight <= 1.0, msg)
            self.assertTrue(g.get_connections()[0].expressed, msg)
            self.assertNotEqual(g.get_connections()[0].in_node, g.get_connections()[0].out_node)
            in_type = g.get_node(g.get_connections()[0].in_node).type
            out_type = g.get_node(g.get_connections()[0].out_node).type
            self.assertFalse(in_type == out_type != 'hidden')
            self.assertFalse((in_type == 'output' and out_type == 'input'))
            self.assertFalse((in_type == 'hidden' and out_type == 'input'))

        # Test to make sure connections are always added (unless at max)
        msg = 'Connection not added!'
        for i in range(2, 4):
            g.mutate_add_connection()
            self.assertEqual(len(g.get_connections()), i, msg)

        # Test to make sure it doesn't go above the maximum connections
        msg = 'Connections exceeded maximum amount!'
        g.mutate_add_connection()
        self.assertEqual(len(g.get_connections()), 4, msg)  # Shouldn't go past 4

    def test_mutate_add_node(self):
        g = Genome(1, 1)

        # Test to make sure you can't add a node without an existing connection
        msg = 'Node added without existing connection!'
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 2, msg)

        # Test adding a node
        msg = 'Node added incorrectly!'
        g.mutate_add_connection()
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 3, msg)
        self.assertEqual(g.get_nodes()[2].id, 2, msg)
        self.assertEqual(g.get_nodes()[2].type, 'hidden', msg)
        self.assertEqual(len(g.get_connections()), 3, msg)
        self.assertEqual(g.get_connections()[0].weight, g.get_connections()[1].weight, msg)
        self.assertEqual(g.get_connections()[2].weight, 1.0, msg)

        # Test to make sure you can't add a node without any expressed connections
        msg = "Node added to disabled connection!"
        for c in g.get_connections():
            c.disable()
        g.mutate_add_node()
        self.assertEqual(len(g.get_nodes()), 3, msg)

    def test_mutate_random_weight(self):
        g = Genome(1, 1)

        msg = 'Failed to set random connection weight!'

        # Test with one connection
        g.add_connection(0, 1, weight=5)
        g.mutate_random_weight()
        self.assertNotEqual(g.get_connections()[0].weight, 5, msg)
        self.assertTrue(-1.0, g.get_connections()[0].weight <= 1.0)

        # Test with multiple connections
        g.add_node(0)
        before = [c.weight for c in g.get_connections()]
        g.mutate_random_weight()
        after = [c.weight for c in g.get_connections()]
        self.assertNotEqual(before, after, msg)

    def test_mutate_shift_weight(self):
        g = Genome(1, 1)
        steps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0]
        error_margin = 0.000000000001

        msg = 'Failed to shift weight correctly!'

        # Test with one connection
        g.add_connection(0, 1, weight=0.0)
        for step in steps:
            current_weight = g.get_connections()[0].weight
            g.mutate_shift_weight(step=step)
            self.assertNotEqual(g.get_connections()[0].weight, current_weight, msg)
            self.assertAlmostEqual(abs(g.get_connections()[0].weight - current_weight), step, delta=error_margin)

        # Test with multiple connections
        g.add_node(0)
        for step in steps:
            before = [c.weight for c in g.get_connections()]
            g.mutate_shift_weight(step=step)
            after = [c.weight for c in g.get_connections()]
            self.assertNotEqual(before, after, msg)

    def test_mutate_toggle_connection(self):
        g = Genome(1, 1)

        msg = 'Failed to toggle connection!'

        # Test with one connection
        g.add_connection(0, 1)
        g.mutate_toggle_connection()
        self.assertFalse(g.get_connections()[0].expressed, msg)
        g.mutate_toggle_connection()
        self.assertTrue(g.get_connections()[0].expressed, msg)

        # Test with multiple connections
        g.add_node(0)
        before = [c.expressed for c in g.get_connections()]
        g.mutate_toggle_connection()
        after = [c.expressed for c in g.get_connections()]
        self.assertNotEqual(before, after, msg)

    def test_set_activation(self):
        g = Genome(3, 3)
        node_ids = range(len(g.get_nodes()))
        act = [activations.relu, activations.gaussian, activations.square]

        # Test to make sure the activation is set correctly
        msg = 'Activation set incorrectly!'
        for nid in node_ids:
            for a in act:
                g.set_activation(nid, a)
                self.assertEqual(g.get_node(nid).activation, a)

    def test_set_connections(self):
        g = Genome(2, 2)

        # Test to make sure connections are set correctly
        msg = 'Connections set incorrectly!'
        connections = [Connection(0, 0, 2), Connection(1, 0, 3), Connection(2, 1, 2), Connection(3, 1, 3)]
        g.set_connections(connections)
        self.assertEqual(g.get_connections(), connections, msg)

        # Make sure weights were compiled correctly
        msg = 'Weights compiled incorrectly!'
        self.assertEqual(g.weights.shape[0], len(g.get_nodes()) - g.shape[0], msg=msg)
        self.assertEqual(g.weights.shape[1], len(g.get_nodes()), msg=msg)
        for i in range(len(g.weights)):
            for j in range(len(g.weights)):
                nid = g._Genome__order[i]
                conn = g.get_connection_from_nodes(j, nid)
                if conn is not None:
                    self.assertEqual(conn.weight, g.weights[i, j], msg)
                else:
                    self.assertEqual(g.weights[i, j], 0.0, msg)

    def test_set_nodes(self):
        g = Genome(1, 1)

        # Test to make sure nodes are set correctly
        msg = 'Nodes set incorrectly!'
        nodes = [Node(0, 'input'), Node(1, 'input'), Node(2, 'output'), Node(3, 'output'), Node(4, 'hidden')]
        g.set_nodes(nodes)
        self.assertEqual(g.get_nodes(), nodes, msg)

        # Make sure weights were compiled correctly
        msg = 'Weights compiled incorrectly!'
        self.assertEqual(g.weights.shape[0], len(nodes) - g.shape[0], msg=msg)
        self.assertEqual(g.weights.shape[1], len(nodes), msg=msg)


class TestNode(unittest.TestCase):
    def test_constructor(self):
        # Test node attribute values
        msg = 'Failed to assign node attributes correctly!'
        n = Node(0, 'input')
        self.assertEqual(n.id, 0, msg)
        self.assertEqual(n.type, 'input', msg)
        self.assertEqual(n.activation, activations.linear, msg)

        n2 = Node(1, 'output')
        self.assertEqual(n2.id, 1, msg)
        self.assertEqual(n2.type, 'output', msg)
        self.assertEqual(n2.activation, activations.modified_sigmoid, msg)

        n3 = Node(2, 'hidden', activation=activations.absolute)
        self.assertEqual(n3.id, 2, msg)
        self.assertEqual(n3.type, 'hidden', msg)
        self.assertEqual(n3.activation, activations.absolute, msg)

    def test_copy(self):
        n = Node(0, 'hidden')
        nc = n.copy()

        # Test to make sure the copies are the same
        msg = 'Copy is not the same as the original!'
        self.assertEqual(n, nc, msg)
        self.assertEqual(n.id, nc.id, msg)
        self.assertEqual(n.type, nc.type, msg)
        self.assertEqual(n.activation, nc.activation, msg)

        # Test to make sure the copy doesn't change if the original does
        msg = 'Copy changed when the original was changed!'
        n.activation = activations.absolute
        self.assertNotEqual(n.activation, nc.activation, msg)

    def test_get_innovation_number(self):
        g = Genome(10, 10)

        msg = 'Innovation number not unique!'

        # Make sure innovation numbers are unique
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                g.add_connection(i, j)
                self.assertEqual(g.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1

        # Make sure it's the same twice
        inn_num = 0
        for i in range(10):
            for j in range(10, 20):
                self.assertEqual(g.get_innovation_number(i, j), inn_num, msg)
                inn_num += 1


class TestConnection(unittest.TestCase):
    def test_constructor(self):
        # Test connection attribute values
        msg = 'Failed to assign connection attributes correctly!'
        c = Connection(0, 0, 2)
        self.assertEqual(c.innovation_number, 0, msg)
        self.assertEqual(c.in_node, 0, msg)
        self.assertEqual(c.out_node, 2, msg)
        self.assertTrue(-1.0 <= c.weight <= 1.0, msg)
        self.assertTrue(c.expressed, msg)

        c2 = Connection(1, 57, 9, weight=-0.4, expressed=False)
        self.assertEqual(c2.innovation_number, 1, msg)
        self.assertEqual(c2.in_node, 57, msg)
        self.assertEqual(c2.out_node, 9, msg)
        self.assertEqual(c2.weight, -0.4, msg)
        self.assertFalse(c2.expressed, msg)

    def test_copy(self):
        c = Connection(0, 0, 2)
        cc = c.copy()

        # Test to make sure the copies are the same
        msg = 'Copy is not the same as the original!'
        self.assertEqual(c, cc, msg)
        self.assertEqual(c.innovation_number, cc.innovation_number, msg)
        self.assertEqual(c.in_node, cc.in_node, msg)
        self.assertEqual(c.out_node, cc.out_node, msg)
        self.assertEqual(c.weight, cc.weight, msg)
        self.assertEqual(c.expressed, cc.expressed, msg)

        # Test to make sure the copy doesn't change if the original does
        msg = 'Copy changed when the original was changed!'
        c.weight = 50
        c.disable()
        self.assertNotEqual(c.weight, cc.weight, msg)
        self.assertNotEqual(c.expressed, cc.expressed, msg)

    def test_disable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to disable connection correctly!'
        c = Connection(0, 0, 2)
        c.disable()
        self.assertFalse(c.expressed, msg)

    def test_enable(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to enable connection correctly'
        c = Connection(0, 0, 2, expressed=False)
        c.enable()
        self.assertTrue(c.expressed, msg)

    def test_set_weight(self):
        # Test to make sure it returns the correct value
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2)
        c.weight = 117
        self.assertEqual(c.weight, 117, msg)

    def test_set_random_weight(self):
        # Test to make sure it returns a random value between -1 and 1
        msg = 'Failed to set connection weight correctly!'
        c = Connection(0, 0, 2, weight=5)
        c.set_random_weight()
        self.assertNotEqual(c.weight, 5, msg)
        self.assertTrue(-1.0 <= c.weight <= 1.0, msg)

    def test_toggle(self):
        # Test to make sure the connection's expression toggles correctly
        msg = 'Failed to toggle connection\'s expression correctly!'
        c = Connection(0, 0, 2)
        c.toggle()
        self.assertFalse(c.expressed, msg)
        c.toggle()
        self.assertTrue(c.expressed, msg)
