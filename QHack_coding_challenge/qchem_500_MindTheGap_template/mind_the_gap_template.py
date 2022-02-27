import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=4)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    @qml.qnode(dev)
    def circuit(parameters):
        # Prepare the HF state: |1100>
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)        
        qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
        qml.SingleExcitation(parameters[1], wires=[0, 2])
        qml.SingleExcitation(parameters[2], wires=[1, 3])

        return qml.expval(H) 
        
    @qml.qnode(dev)        
    def circuit_state(parameters):
        # Prepare the HF state: |1100>
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)  
        qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
        qml.SingleExcitation(parameters[1], wires=[0, 2])
        qml.SingleExcitation(parameters[2], wires=[1, 3])

        return qml.state() 
    params = np.zeros(3, requires_grad=True)

    prev_energy = 0.0
    for n in range(50):
        # perform optimization step
        params, energy = opt.step_and_cost(circuit, params)

        if np.abs(energy - prev_energy) < 1e-6:
            break
        prev_energy = energy

    # store the converged parameters
    state_result = circuit_state(params)
    energy_result = energy
    return energy_result, params, state_result
    # QHACK #





def excited_state_VQE(H, ground_params, ground_state):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """
    def create_H1(ground_state, beta, H):
        """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

        Args:
            - ground_state (np.ndarray): from the ground state VQE calculation
            - beta (float): the prefactor for the ground state projector term
            - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

        Returns:
            - (qml.Observable): The result of qml.Hermitian(H1_matrix)
        """

        # QHACK #
        H = qml.utils.sparse_hamiltonian(H).toarray()
        #print(np.linalg.eigvals(H))
        penalty = beta*np.outer(ground_state, ground_state)
        H_hat = H + penalty    
        #print(np.linalg.eigvals(H_hat)) - works OK.... somehow
        result = qml.Hermitian(H_hat,[0,1,2,3])
        return result
    
    
    # QHACK #
    H1 = create_H1(ground_state, 15, H)
    H2 = create_H1(ground_state, 15, H)
    H3 = create_H1(ground_state, 10, H)
    dev = qml.device("default.qubit", wires=4)
    #annealing
    opt_0 = qml.RMSPropOptimizer(stepsize=0.01, decay=0.9, eps=1e-08)
    opt_1 = qml.AdamOptimizer(stepsize=0.4)
    opt_2 = qml.AdamOptimizer(stepsize=0.1)
    @qml.qnode(dev)
    def circuit_H1(parameters):
        # Prepare the HF state: |1100>
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)        
        qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
        qml.SingleExcitation(parameters[1], wires=[0, 2])
        qml.SingleExcitation(parameters[2], wires=[1, 3])

        return qml.expval(H1) 
    #@qml.qnode(dev)
    #def circuit_H2(parameters):
    #    # Prepare the HF state: |1100>
    #    qml.PauliX(wires=0)
    #    qml.PauliX(wires=1)        
    #    qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
    #    qml.SingleExcitation(parameters[1], wires=[0, 2])
    #    qml.SingleExcitation(parameters[2], wires=[1, 3])

    #    return qml.expval(H2) 
    #@qml.qnode(dev)
    #def circuit_H3(parameters):
    #    # Prepare the HF state: |1100>
    #    qml.PauliX(wires=0)
    #    qml.PauliX(wires=1)        
    #    qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
    #    qml.SingleExcitation(parameters[1], wires=[0, 2])
    #    qml.SingleExcitation(parameters[2], wires=[1, 3])

    #    return qml.expval(H3)         
    @qml.qnode(dev)        
    def circuit_E(parameters):
        # Prepare the HF state: |1100>
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)  
        qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])
        qml.SingleExcitation(parameters[1], wires=[0, 2])
        qml.SingleExcitation(parameters[2], wires=[1, 3])

        return qml.expval(H)  
    params = np.zeros(3, requires_grad=True)
    for i in range(1,3):
        params[i] = np.pi/4 #ground_params[i]
    prev_energy = 15.0    
    for n in range(200):
        # perform optimization step
        params, energy = opt_0.step_and_cost(circuit_H1, params)
        e_old = circuit_E(params)
        #if np.abs(energy - prev_energy) < 1e-6:
        #    break
        #if n % 10 == 0:
        #    print(n, energy, e_old)
        #if energy < prev_energy:        
        #    prev_energy = energy
            
    #prev_energy = 15.0                 
    #for i in range(3):
    #    params[i] = best_param[i]        
    #for n in range(400):
        # perform optimization step
        #params, energy = opt_1.step_and_cost(circuit_H2, params)
        #e_old = circuit_E(params)
        #if np.abs(energy - prev_energy) < 1e-6:
        #    break
        #if n % 10 == 0:
        #    print(n, energy, e_old)
            
    # store the converged parameters
    energy_result = circuit_E(params)
    return energy_result
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, params, ground_state = ground_state_VQE(H)

    beta = 15.0
    #H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H, params, ground_state)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
