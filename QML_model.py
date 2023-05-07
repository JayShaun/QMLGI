

import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import struct
import os


K = tc.set_backend("tensorflow")

Rx = tc.gates.rx
Ry = tc.gates.ry
Rz = tc.gates.rz

def entangling_layer(c, qubits):
    """
    qubits: [0,1,2,3,4..] qubit index list
    Return a layer of CZ entangling gates on "qubits" (arranged in a circular topology)
    """
    for q0, q1 in zip(qubits, qubits[1:]):  
        c.cz(q0,q1) 

    if len(qubits)!=2:   
        c.cz(qubits[0], qubits[-1])

def inputs_encoding(c, input_params):
    """
    params = np.array(size=(n_qubit,)) 
    here the input_params are scaled input and scaling will be put into the torch layer
    """
    for qubit, input in enumerate(input_params):
        c.rx(qubit, input)  

def entangling_layer_trainable_noise(c, qubits, symbols, px, py, pz, seed):
    """
    Returns a layer of ZZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    moreover the decoherence operator is followed by the ZZ gates.
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary = tc.gates._zz_matrix)
        c.depolarizing(q0, px=px, py=py, pz=pz, status=seed[i, 0]) 
        c.depolarizing(q1, px=px, py=py, pz=pz, status=seed[i, 1])
    
    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary = tc.gates._zz_matrix)
        c.depolarizing(qubits[0], px=px, py=py, pz=pz, status=seed[-1, 0]) 
        c.depolarizing(qubits[-1], px=px, py=py, pz=pz, status=seed[-1, 1])

def entangling_layer_trainable(c, qubits, symbols):
    """
    Returns a layer of ZZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary = tc.gates._zz_matrix)

    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary = tc.gates._zz_matrix)
      

def quantum_circuit(inputs, params, lamda_scaling, q_weights_final):
    """
    Prepares a data re-uploading circuit on "qubits" with 'n_layers' layers 
    inputs: len(inputs) = n_qubits
    weights.shape (n_layers+1, n_qubits, 3) as each qubit has three parameters to be encoded
    Only weights are trainable
    lambda_scaling: (n_qubits*n_layers, )
    """

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
   
    qubits = range(n_qubits)
   
    for qubit in qubits:
        c.H(qubit)   
    
    for layer in range(n_layers):  
        # Variational layer
        for qubit in range(n_qubits): 
            c.ry(qubit, theta=inputs[qubit]*lamda_scaling[layer, qubit])

        for qubit in qubits:    
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

       
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits

    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])
    return outputs

def quantum_circuit_fixing(inputs, params, q_weights_final):
    """
    Prepares a data re-uploading circuit on "qubits" with 'n_layers' layers 
    inputs: len(inputs) = n_qubits
    weights.shape (n_layers+1, n_qubits, 3) as each qubit has three parameters to be encoded
    Only weights are trainable
    lambda_scaling: (n_qubits*n_layers, )
    """

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)

    qubits = range(n_qubits)
    
    for qubit in qubits:
        c.H(qubit)   
    for layer in range(n_layers):  
        # Variational layer
        for qubit in range(n_qubits): 
            c.ry(qubit, theta=inputs[qubit])

        for qubit in qubits:   
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        entangling_layer_trainable(c, qubits, weight_ent[layer, :])
      
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])
    return outputs

def quantum_circuit_fixing_wout(inputs, params, q_weights_final, w_out):
    

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    
   
    qubits = range(n_qubits)
    
    for qubit in qubits:
        c.H(qubit)   
    
    for layer in range(n_layers):  
        
        for qubit in range(n_qubits): 
            c.ry(qubit, theta=inputs[qubit])

        for qubit in qubits:   
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits
      
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0]) 
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1]) * w_out
    return outputs

def quantum_circuit_simple(inputs, params, q_weights_final):
   
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]

    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
   
    qubits = range(n_qubits)
    
    for qubit in range(n_qubits): ## single data encoding
        c.ry(qubit, theta=inputs[qubit])

    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        
        entangling_layer_trainable(c, qubits, weights_ent[layer, :]) 

    
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
       
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(int(n_qubits/2))]
        + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(int(n_qubits/2), n_qubits)] 
    )
    outputs = K.reshape(outputs, [-1])
    return outputs


def quantum_circuit_Noise(inputs, params, q_weights_final):
    """
    Adding the noise into the quantum circuit followed by the tow qubit gates.
    """
   
    px, py, pz = 0.05, 0.05, 0.05  
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]
    
    n_layers, n_qubits, _ = weights_var.shape
    input_data = inputs[:n_qubits]
    seeds = inputs[n_qubits:]
    seeds = K.reshape(seeds, (n_layers, n_qubits, 2))
    
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    for layer in range(n_layers): 
        
        for qubit in range(n_qubits): 
            c.ry(qubit, theta=input_data[qubit])
   
        for qubit in qubits:   
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        entangling_layer_trainable_noise(c, qubits, weights_ent[layer, :], px,py,pz, seeds[layer,]) 

    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )
  
    outputs = K.reshape(outputs, [-1])

    return outputs


def encoding_IQP(c, inputs, qubits):
    # inputs = 2*np.pi*inputs
    ## encoding layer: IQP 
    for i in qubits:
        c.H(i)
        c.rz(i, theta=inputs[i])

    for q0, q1 in zip(qubits, qubits[1:]):
       
        c.exp1(q0, q1, theta=inputs[q0]*inputs[q1], unitary = tc.gates._zz_matrix)

    for i in qubits:
        c.H(i)
        c.rz(i, theta=inputs[i])
    
    for q0, q1 in zip(qubits, qubits[1:]):
        
        c.exp1(q0, q1, theta=inputs[q0]*inputs[q1], unitary = tc.gates._zz_matrix)

def quantum_IQP(inputs, params, q_weights_final):
    """
    IQP encoding:
    exp(\sum_i x_i Z_i + \sum_j \sum_j' x_ij x_ij' Z_j Z_j' )
    """
    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    encoding_IQP(c, inputs, qubits)

    
    for layer in range(n_layers):  

       
        for qubit in qubits:   
            
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

       
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

   
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )

    outputs = K.reshape(outputs, [-1])
    return outputs

from scipy.stats import unitary_group
def Haar(i):
    np.random.seed(42+i)
    haar_matrix = unitary_group.rvs(2, random_state=42+i)
    return haar_matrix.astype(np.complex64)

def quantum_Heisenberg(inputs, params, q_weights_final):
    """
    define a quantum Heisenberg model to encode the quantum circuit
    for 16 elements in inputs, there are 17 qubits to encoding.
    The basic time step is 
    \prod_j=1^n exp(-i*t/T*x_j(X_jX_j+1+Y_jY_j+1+Z_jZ_j+1))
    """

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]

    n_layers, n_qubits, _ = weights_var.shape
    
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    T=1
    t=float(3/(n_qubits-1))
    
    for qubit in qubits:
       
        c.any(qubit, unitary=Haar(qubit))
    for _ in range(T):
       
        for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
            c.RXX(q0,q1,theta=inputs[i]*(t/T))
            c.RYY(q0,q1,theta=inputs[i]*(t/T))
            c.RZZ(q0,q1,theta=inputs[i]*(t/T))
          

    
    for layer in range(n_layers):  
      
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

       
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

    
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
   
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
    )

    outputs = K.reshape(outputs, [-1])
    return outputs
    




