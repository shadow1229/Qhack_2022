#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """

    @qml.qnode(dev)
    def circuit1():
        # QHACK #
        qml.PauliX(wires='sol')
        qml.Hadamard(wires=0)
        qml.Hadamard(wires='sol')       
        oracle()
        qml.Hadamard(wires=0)   
        # QHACK #
        return qml.sample()

    @qml.qnode(dev)
    def circuit2(wire_1):
        # QHACK #
        if wire_1 == 1:
            qml.PauliX(wires=1)
        oracle()
        # QHACK #
        return qml.sample()

    sol1 = circuit1()
    if sol1[0] == 1:
        wire_1 = 0
    else:
        wire_1 = 1
        
    sol2 = circuit2(wire_1)
    if sol2[2] == 1:
        wire_0 = 0
    else:
        wire_0 = 1

    result = 2*wire_0+wire_1
    return result
    # QHACK #

    # process sol1 and sol2 to determine which door the car is behind.

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)
        qml.Toffoli(wires=[0, 1, "sol"])
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)

    output = find_the_car(oracle)
    print(f"{output}")
