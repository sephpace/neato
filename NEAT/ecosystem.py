
from genome import Genome


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
