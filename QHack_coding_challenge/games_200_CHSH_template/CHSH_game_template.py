#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    #done
    theta = np.arctan2(beta,alpha) * 2
    qml.RY(theta, wires= 0) 
    qml.CNOT( wires = [0,1]) 
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)
    
    if x == 0:
        qml.RY(2*theta_A0, wires = 0)
    else:
        qml.RY(2*theta_A1, wires = 0)
        
    if y == 0:
        qml.RY(2*theta_B0, wires = 1)
    else:
        qml.RY(2*theta_B1, wires = 1)

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """
    # QHACK #
    win_sum = 0.0
    for x in range(2):
        for y in range(2):
            probs = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
            if x == 1 and y == 1:
                win_sum += probs[1] + probs[2]
            else:
                win_sum += probs[0] + probs[3]
                
    return win_sum / 4.0
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """
    
    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        prob = winning_prob(params, alpha, beta)
        return -1*prob

    # QHACK #
    
    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.ones(4, requires_grad=True) * (np.pi/4)
    opt = qml.AdagradOptimizer(stepsize=0.3)

    # QHACK #
    
    # set the initial parameter values
    params = init_params
    epochs = 1200
    
    prob_best = 0
    param_best = [0,0,0,0]
    for epoch in range(epochs):
        params = opt.step(cost, params) 
        prob = -1 * cost(params)
        if prob > prob_best:
            prob_prev = prob
            param_best[0] = params[0]
            param_best[1] = params[1]
            param_best[2] = params[2]
            param_best[3] = params[3]            
        
    return winning_prob(param_best, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
