#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml






def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    
    # QHACK #
    dev = qml.device("default.qubit", wires=7, shots=1)

    @qml.qnode(dev)
    def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #

        # Insert any pre-oracle processing here
        qml.PauliX(wires = 2 )
        for i in range(3):
            qml.Hadamard(wires = i)

        fs[0](wires=range(7))

        # Insert any post-oracle processing here
        for i in range(2):
            qml.Hadamard(wires = i)
        
        #sorry pennylane team... I think I'm super idiot....
        #prepare for the classical way!             
        qml.SWAP(wires=[0,3])
        qml.SWAP(wires=[1,4])

        # Insert any pre-oracle processing here
        for i in range(2):
            qml.Hadamard(wires = i)

        fs[1](wires=range(7))

        # Insert any post-oracle processing here
        for i in range(2):
            qml.Hadamard(wires = i)        
        qml.SWAP(wires=[0,5])
        qml.SWAP(wires=[1,6])
        
        # Insert any pre-oracle processing here
        for i in range(2):
            qml.Hadamard(wires = i)

        fs[2](wires=range(7))

        # Insert any post-oracle processing here
        for i in range(2):
            qml.Hadamard(wires = i)        
        # QHACK #
        
        return qml.sample(wires=range(7))
    #drawer = qml.draw(circuit)
    #print(drawer())
    sample = circuit()
    #print(sample)
    fs_0 = (sample[3]+sample[4]) == 0
    fs_1 = (sample[5]+sample[6]) == 0
    fs_2 = (sample[0]+sample[1]) == 0
    #print(fs_0, fs_1, fs_2)
    if fs_0 == fs_1 and fs_0 == fs_2:
        return "4 same"
    else:
        return "2 and 2"
    
    #sorry again!
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
