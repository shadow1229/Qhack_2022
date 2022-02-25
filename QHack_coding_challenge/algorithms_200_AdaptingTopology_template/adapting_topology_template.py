#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    wires = cnot.wires
    start = wires[0]
    end = wires[1]
    
    least_n =[9999 for i in range(9)]
    
    least_n[start] = 0
    is_change = True
    while is_change:
        is_change = False
        for i in range(9):
            if least_n[i] == 9999:
                continue
                
            new_n = least_n[i] + 1
            for v in graph[i]:
                if new_n < least_n[v]:
                    is_change = True
                    least_n[v] = new_n
                     
    return 2*(least_n[end]-1)
        
        
    
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
