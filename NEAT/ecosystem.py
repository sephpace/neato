
import random

from genome import Genome


class Ecosystem:
    def __init__(self):
        self.__inn_num_gen = innovation_number_generator()
        self.__inn_num_gen.send(None)
        self.__genomes = []

    def create_genome(self, input_amt, output_amt):
        self.__genomes.append(Genome(input_amt, output_amt, self.__inn_num_gen))

    def cross(self, genome1, genome2):
        """
        Returns a combination of the two given genomes.

        Parameters:
        genome1 (Genome): The first genome to cross
        genome2 (Genome): The second genome to cross

        Returns:
        (Genome): A genome that is a combination of the two given genomes
        """
        # Find the fitter parent
        if genome1.get_fitness() >= genome2.get_fitness():
            more_fit_parent = genome1
            less_fit_parent = genome2
        else:
            more_fit_parent = genome2
            less_fit_parent = genome1

        # Combine the genomes
        more_fit_connections = sorted([c.get_innovation_number() for c in more_fit_parent.get_connections()])
        less_fit_connections = sorted([c.get_innovation_number() for c in less_fit_parent.get_connections()])

        child_nodes = []
        child_connections = []

        # TODO: Find a way to make this look less messy

        for i in range(max(max(more_fit_connections), max(less_fit_connections)) + 1):
            # --- Matching gene for both parents ---
            if i in more_fit_connections and i in less_fit_connections:
                rand_choice = random.randint(0, 1)

                # More fit parent
                if rand_choice == 0:
                    conn = more_fit_parent.get_connection(i)

                    if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(more_fit_parent.get_node(conn.get_in_node()))

                    if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(more_fit_parent.get_node(conn.get_out_node()))

                # Less fit parent
                else:
                    conn = less_fit_parent.get_connection(i)

                    if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(less_fit_parent.get_node(conn.get_in_node()))

                    if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                        child_nodes.append(less_fit_parent.get_node(conn.get_out_node()))

                child_connections.append(conn)

            # --- Disjoint or excess gene for fit parent ---
            elif i in more_fit_connections and i not in less_fit_connections:
                conn = more_fit_parent.get_connection(i)

                if conn.get_in_node() not in [n.get_id() for n in child_nodes]:
                    child_nodes.append(more_fit_parent.get_node(conn.get_in_node()))

                if conn.get_out_node() not in [n.get_id() for n in child_nodes]:
                    child_nodes.append(more_fit_parent.get_node(conn.get_out_node()))

                child_connections.append(conn)

            # --- Disjoint or excess for less fit parent or gene doesn't exist in either parent ---
            else:
                continue

        # Sort the nodes and connections
        child_nodes.sort(key=lambda node: node.get_id())
        child_connections.sort(key=lambda conn: conn.get_innovation_number())

        # Create the child
        child = Genome(0, 0, self.__inn_num_gen)
        child.set_nodes(child_nodes)
        child.set_connections(child_connections)

        return child

    def get_genomes(self): return self.__genomes


def innovation_number_generator():
    """
    A generator that given the in node and out node of a connection as a tuple through the send function,
    yields a unique innovation number for that connection.  If the connection already exists within the
    ecosystem, the existing innovation number is yielded.

    The generator function send(None) must be called after declaration.
    """
    inn_log = []
    inn_num = 0
    while True:
        conn = yield inn_num
        if conn in inn_log:
            inn_num = inn_log.index(conn)
        else:
            inn_log.append(conn)
            inn_num = len(inn_log) - 1
