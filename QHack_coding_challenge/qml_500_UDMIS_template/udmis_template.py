import sys
import pennylane as qml
from pennylane import numpy as np


def hamiltonian_coeffs_and_obs(graph):
    """Creates an ordered list of coefficients and observables used to construct
    the UDMIS Hamiltonian.

    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]

    Returns:
        - coeffs (list): List of coefficients for elementary parts of the UDMIS Hamiltonian
        - obs (list(qml.ops)): List of qml.ops
    """

    num_vertices = len(graph)
    E, num_edges = edges(graph)
    u = 1.35
    obs = []
    coeffs = []

    obs.append(qml.Identity(0))
    coeffs.append(0)  
          
    for v in range(num_vertices):
        #-(Pauliz + 1 ) /2
        obs.append(qml.PauliZ(v))
        coeffs.append(-0.5)
        coeffs[0] -= 0.5 #identity
    
    # u* ((Pauliz1 + 1 ) /2 ) *  ((Pauliz2 + 1 ) /2 ) = 0.25*u *z1 @ z2 + 0.25u * z1 + 0.25u * z2 + 0.25u I 
    for vertex_i in range(num_vertices - 1):
        for vertex_j in range(vertex_i + 1, num_vertices):
            if E[vertex_i][vertex_j] == True:
                obs.append(qml.PauliZ(vertex_i) @ qml.PauliZ(vertex_j) )
                coeffs.append(0.25*u) 
                coeffs[0] += 0.25*u #identity
                coeffs[vertex_i+1] += 0.25*u #pauliz, keep in mind index 0 is identity!
                coeffs[vertex_j+1] += 0.25*u

    # create the Hamiltonian coeffs and obs variables here
    # QHACK #

    return coeffs, obs


def edges(graph):
    """Creates a matrix of bools that are interpreted as the existence/non-existence (True/False)
    of edges between vertices (i,j).

    Args:
        - graph (list((float, float))): A list of x,y coordinates. e.g. graph = [(1.0, 1.1), (4.5, 3.1)]

    Returns:
        - num_edges (int): The total number of edges in the graph
        - E (np.ndarray): A Matrix of edges
    """

    # DO NOT MODIFY anything in this code block
    num_vertices = len(graph)
    E = np.zeros((num_vertices, num_vertices), dtype=bool)
    for vertex_i in range(num_vertices - 1):
        xi, yi = graph[vertex_i]  # coordinates

        for vertex_j in range(vertex_i + 1, num_vertices):
            xj, yj = graph[vertex_j]  # coordinates
            dij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            E[vertex_i, vertex_j] = 1 if dij <= 1.0 else 0

    return E, np.sum(E, axis=(0, 1))


def variational_circuit(params, num_vertices, graph):
    """A variational circuit.

    Args:
        - params (np.ndarray): your variational parameters
        - num_vertices (int): The number of vertices in the graph. Also used for number of wires.
    """

    # QHACK #

    # create your variational circuit here
    #
    E, num_edges = edges(graph)
    
    edge_list = []
    for vertex_i in range(num_vertices - 1):
        for vertex_j in range(vertex_i + 1, num_vertices):
            if E[vertex_i][vertex_j] == True:
                edge_list.append([vertex_i, vertex_j])    
    
    for v in range(num_vertices):
        qml.RX(params[v,0], wires=v)  #this should work in example 1, where edge is not exists
        qml.RZ(params[v,1], wires=v)  #this should work in example 1, where edge is not exists
        #add this if training for example 2 is not working well
        #qml.SingleExcitation(parameters[1], wires=[0, 2]) #
        
    for i, e in enumerate(edge_list):
        qml.SingleExcitation(params[(num_vertices+i),0], wires=[e[0], e[1]]) 
    for i, e in enumerate(edge_list):
        qml.CRY(params[(num_vertices+i),1], wires=[e[0], e[1]]) 
    # QHACK #


def train_circuit(num_vertices, H, graph):
    """Trains a quantum circuit to learn the ground state of the UDMIS Hamiltonian.

    Args:
        - num_vertices (int): The number of vertices/wires in the graph
        - H (qml.Hamiltonian): The result of qml.Hamiltonian(coeffs, obs)

    Returns:
        - E / num_vertices (float): The ground state energy density.
    """
    edge_matrix, num_edges = edges(graph)
    dev = qml.device("default.qubit", wires=num_vertices)

    @qml.qnode(dev)
    def cost(params):
        """The energy expectation value of a Hamiltonian"""
        variational_circuit(params, num_vertices, graph)
        return qml.expval(H)

    # QHACK #

    # define your trainable parameters and optimizer here
    # change the number of training iterations, `epochs`, if you want to
    # just be aware of the 80s time limit!
    E = 999.9
    epochs = 500
    params = 2*np.pi * np.random.rand((num_vertices+num_edges), 2, requires_grad=True)
    # QHACK #
    opt = qml.RMSPropOptimizer(stepsize=0.01, decay=0.9, eps=1e-08)
    for i in range(epochs):
        params, E = opt.step_and_cost(cost, params)
        #print( i, E / float(num_vertices) )
    return E / float(num_vertices)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float, requires_grad=False)
    num_vertices = int(len(inputs) / 2)
    x = inputs[:num_vertices]
    y = inputs[num_vertices:]
    graph = []
    for n in range(num_vertices):
        graph.append((x[n].item(), y[n].item()))

    coeffs, obs = hamiltonian_coeffs_and_obs(graph)
    H = qml.Hamiltonian(coeffs, obs)

    energy_density = train_circuit(num_vertices, H, graph)
    print(f"{energy_density:.6f}")
