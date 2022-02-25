#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)        
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        cnt_list = ['000','001','010','011','100','101','110','111']
        u_list = [ np.zeros((2,2)) for i in range(8)]
        for i in range(8):
            theta = thetas[i]
            u_list[i][0][0] = np.cos(theta/2)
            u_list[i][0][1] =-np.sin(theta/2)
            u_list[i][1][0] = np.sin(theta/2)
            u_list[i][1][1] = np.cos(theta/2)

        for i in range(8):
            qml.ControlledQubitUnitary(u_list[i], control_wires=[0,1,2], wires=3, control_values=cnt_list[i])
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
