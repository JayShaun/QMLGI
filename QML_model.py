

import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import struct
import os
# from tensorcircuit import keras
# from tensorflow import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K = tc.set_backend("tensorflow")

Rx = tc.gates.rx
Ry = tc.gates.ry
Rz = tc.gates.rz

def entangling_layer(c, qubits):
    """
    qubits: [0,1,2,3,4..] qubit index list
    Return a layer of CZ entangling gates on "qubits" (arranged in a circular topology)
    """
    for q0, q1 in zip(qubits, qubits[1:]):  ## CNOT and CZ can both be well
        c.cz(q0,q1)  ## 第五个qubits不打包了。这是zip的特性

    if len(qubits)!=2:   ## 超过2个qubits，那么就在第一个和最后一个上面加入CZ门
        c.cz(qubits[0], qubits[-1])

def inputs_encoding(c, input_params):
    """
    params = np.array(size=(n_qubit,)) 
    here the input_params are scaled input and scaling will be put into the torch layer
    """
    for qubit, input in enumerate(input_params):
        c.rx(qubit, input)  ## 直接rx就好了 所以每个scaling 

def entangling_layer_trainable_noise(c, qubits, symbols, px, py, pz, seed):
    """
    Returns a layer of ZZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    moreover the decoherence operator is followed by the ZZ gates.
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary = tc.gates._zz_matrix)
        c.depolarizing(q0, px=px, py=py, pz=pz, status=seed[i, 0]) ## 加入noise
        c.depolarizing(q1, px=px, py=py, pz=pz, status=seed[i, 1])
    
    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary = tc.gates._zz_matrix)
        c.depolarizing(qubits[0], px=px, py=py, pz=pz, status=seed[-1, 0]) ## 加入noise
        c.depolarizing(qubits[-1], px=px, py=py, pz=pz, status=seed[-1, 1])

def entangling_layer_trainable(c, qubits, symbols):
    """
    Returns a layer of ZZ entangling gates on `qubits` (arranged in a circular topology).
    the length of symbols equals to the number of qubits
    """
    for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        c.exp1(q0, q1, theta=symbols[i], unitary = tc.gates._zz_matrix)
        # c.depolarizing(q0, px=px, py=py, pz=pz, status=seed[i, 0]) ## 加入noise
        # c.depolarizing(q1, px=px, py=py, pz=pz, status=seed[i, 1])
    
    if len(qubits) != 2:
        c.exp1(qubits[0], qubits[-1], theta=symbols[-1], unitary = tc.gates._zz_matrix)
        # c.depolarizing(qubits[0], px=px, py=py, pz=pz, status=seed[-1, 0]) ## 加入noise
        # c.depolarizing(qubits[-1], px=px, py=py, pz=pz, status=seed[-1, 1])

def quantum_circuit(inputs, params, lamda_scaling, q_weights_final):
    """
    Prepares a data re-uploading circuit on "qubits" with 'n_layers' layers 
    inputs: len(inputs) = n_qubits
    weights.shape (n_layers+1, n_qubits, 3) as each qubit has three parameters to be encoded
    Only weights are trainable
    lambda_scaling: (n_qubits*n_layers, )
    这里存在的问题是 batch_input首先会运行一遍circuit，因此我们的inputs是会被scale的，因此inputs就会被认为是trainable.
    """
    #lambda_scaling with shape (layers, qubits)

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    # print(scaled_inputs.shape)
    ## 这里能够保证所有的训练参数都是相同，尽管input经过scaling，但是他还是在不同的batch之间保证了相同
    qubits = range(n_qubits)
    ## the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   ## transform them into two |+> state
    ## 这里的input不对呀，第一个维度是batch，第二个维度是layers, 第三个维度是
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        # Variational layer
        for qubit in range(n_qubits): ## data reuploading 
            c.ry(qubit, theta=inputs[qubit]*lamda_scaling[layer, qubit])
            # c.rz(qubit, theta=inputs[qubit]*lamda_scaling[layer, qubit, 1])

        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            # c.ry(qubit, theta=weights_var[layer,qubit, 1])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits
        # Encoding layer
        # for qubit in qubits:
        # inputs_encoding(inputs[layer, :])  ## 这里会reuploading inputs
    # # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] + 
        #     [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
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
    这里存在的问题是 batch_input首先会运行一遍circuit，因此我们的inputs是会被scale的，因此inputs就会被认为是trainable.
    """
    #lambda_scaling with shape (layers, qubits)

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    
    # print(scaled_inputs.shape)
    ## 这里能够保证所有的训练参数都是相同，尽管input经过scaling，但是他还是在不同的batch之间保证了相同
    qubits = range(n_qubits)
    ## the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   ## transform them into two |+> state
    ## 这里的input不对呀，第一个维度是batch，第二个维度是layers, 第三个维度是
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        # Variational layer
        for qubit in range(n_qubits): ## data reuploading 
            c.ry(qubit, theta=inputs[qubit])

        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            # c.ry(qubit, theta=weights_var[layer,qubit, 1])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits
        # Encoding layer
        # for qubit in qubits:
        # inputs_encoding(inputs[layer, :])  ## 这里会reuploading inputs
    # # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)]
        #     [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])
    return outputs

def quantum_circuit_fixing_wout(inputs, params, q_weights_final, w_out):
    """
    Prepares a data re-uploading circuit on "qubits" with 'n_layers' layers 
    inputs: len(inputs) = n_qubits
    weights.shape (n_layers+1, n_qubits, 3) as each qubit has three parameters to be encoded
    Only weights are trainable
    lambda_scaling: (n_qubits*n_layers, )
    这里存在的问题是 batch_input首先会运行一遍circuit，因此我们的inputs是会被scale的，因此inputs就会被认为是trainable.
    """
    #lambda_scaling with shape (layers, qubits)

    weights_var = params[:,:,:2]
    weight_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    
    # print(scaled_inputs.shape)
    ## 这里能够保证所有的训练参数都是相同，尽管input经过scaling，但是他还是在不同的batch之间保证了相同
    qubits = range(n_qubits)
    ## the first Hardmard layer with no repeating
    for qubit in qubits:
        c.H(qubit)   ## transform them into two |+> state
    ## 这里的input不对呀，第一个维度是batch，第二个维度是layers, 第三个维度是
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        # Variational layer
        for qubit in range(n_qubits): ## data reuploading 
            c.ry(qubit, theta=inputs[qubit])

        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            # c.ry(qubit, theta=weights_var[layer,qubit, 1])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## entangling layer, weight_ent
        entangling_layer_trainable(c, qubits, weight_ent[layer, :]) ## there is no variantional qubits
        # Encoding layer
        # for qubit in qubits:
        # inputs_encoding(inputs[layer, :])  ## 这里会reuploading inputs
    # # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(n_qubits)] + 
        #     [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1]) * w_out
    return outputs

def quantum_circuit_simple(inputs, params, q_weights_final):
    """
    This circuit implements the QCBM with the simple quantum-bits encoding, not reuploading
    The circuit uses half of qubits as the Z measurement and uses another half qubits as x measurements.
    """
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]

    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    # print(scaled_inputs.shape)
    ## 这里能够保证所有的训练参数都是相同，尽管input经过scaling，但是他还是在不同的batch之间保证了相同
    qubits = range(n_qubits)
    ## the first Hardmard layer with no repeating
    # for qubit in qubits:
    #     c.H(qubit)   ## transform them into two |+> state

    for qubit in range(n_qubits): ## single data encoding
        c.ry(qubit, theta=inputs[qubit])

    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable(c, qubits, weights_ent[layer, :]) 

    # # Last varitional layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(int(n_qubits/2))]
        + [K.real(c.expectation([tc.gates.x(), [i]])) for i in range(int(n_qubits/2), n_qubits)] 
        #     +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    outputs = K.reshape(outputs, [-1])
    return outputs

def quantum_circuit_CS(inputs_basis, params, q_weights_final, seed):
    """
    The circuit implements classical shadow implementation
    """
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    inputs = inputs_basis[:n_qubits]
    basis = inputs_basis[n_qubits:]

    
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        
        for qubit in range(n_qubits): ## single data encoding
            c.ry(qubit, theta=inputs[qubit])
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable(c, qubits, weights_ent[layer, :]) 
    
    ## last variational layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    ## use classical shadow
    expectation = tc.templates.measurements.parameterized_measurements(c, basis, onehot=True)
    outputs = K.stack([K.real(expectation)]+[basis])
    outputs = K.reshape(outputs, [-1])

    return outputs

def quantum_circuit_O(inputs, params, q_weights_final):
    """
    The circuit implements trainable basis but the output of the bits are doubled
    trainable_basis is shaped with (qubits,)
    """
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)


    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        
        for qubit in range(n_qubits): ## single data encoding
            c.ry(qubit, theta=inputs[qubit])
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable(c, qubits, weights_ent[layer, :]) 

    ## last variational layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    ## 计算expectation实际上是可以通过shot进行估计的。那是一种更加实际的做法。
    ## 增加Y，orthogonal basis
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        + [K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)] 
        #     +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    ## concanating them into a vetor.
    outputs = K.reshape(outputs, [-1])

    return outputs


def quantum_circuit_TB(inputs, params, q_weights_final,training_basis):
    """
    The circuit implements trainable basis but the output of the bits are doubled
    trainable_basis is shaped with (qubits,)
    """
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]
    n_layers, n_qubits, _ = weights_var.shape
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        
        for qubit in range(n_qubits): ## single data encoding
            c.ry(qubit, theta=inputs[qubit])
        
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable(c, qubits, weights_ent[layer, :]) 

    ## last variational layer
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        # c.ry(qubit, theta=q_weights_final[qubit, 1])
        c.rz(qubit, theta=q_weights_final[qubit, 1])

    ## 计算expectation实际上是可以通过shot进行估计的。那是一种更加实际的做法。
    ## 这里采用rx门来代替z门进行测量，
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(int(n_qubits/2))]
        + [K.real(c.expectation([Rx(training_basis[i]), [i]])) for i in range(int(n_qubits/2), n_qubits)] 
        #     +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    ## concanating them into a vetor.
    outputs = K.reshape(outputs, [-1])
    outputs = (outputs + 1)/2

    return outputs

def quantum_circuit_Noise(inputs, params, q_weights_final):
    """
    Adding the noise into the quantum circuit followed by the tow qubit gates.
    """
    ## CPU 上0.005 
    ## GPU 上0.1
    px, py, pz = 0.05, 0.05, 0.05  # small noise value, noise seriously influence the performance of
    weights_var = params[:,:,:2]
    weights_ent = params[:,:,-1]
    ## 对inputs进行拆分，一部分是noise seed 一部分是data
    n_layers, n_qubits, _ = weights_var.shape
    input_data = inputs[:n_qubits]
    seeds = inputs[n_qubits:]
    seeds = K.reshape(seeds, (n_layers, n_qubits, 2))
    # seeds = K.implicit_randu([n_layers, n_qubits, 2])  not reproduceble
    c = tc.Circuit(n_qubits)
    qubits = range(n_qubits)

    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        
        for qubit in range(n_qubits): ## reuploading data encoding
            c.ry(qubit, theta=input_data[qubit])
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable_noise(c, qubits, weights_ent[layer, :], px,py,pz, seeds[layer,]) 

    ## last variational layer for constructing Haar unitary
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    ## 计算expectation实际上是可以通过shot进行估计的。那是一种更加实际的做法。
    ## 这里采用rx门来代替z门进行测量，
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([Rx(training_basis[i]), [i]])) for i in range(n_qubits)] 
            # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )
    ## concanating them into a vetor.
    outputs = K.reshape(outputs, [-1])

    return outputs

# from itertools import combinations

def encoding_IQP(c, inputs, qubits):
    # inputs = 2*np.pi*inputs
    ## encoding layer: IQP 
    for i in qubits:
        c.H(i)
        c.rz(i, theta=inputs[i])

    for q0, q1 in zip(qubits, qubits[1:]):
        ## IQP circuit可能非常不好实现
        c.exp1(q0, q1, theta=inputs[q0]*inputs[q1], unitary = tc.gates._zz_matrix)

    for i in qubits:
        c.H(i)
        c.rz(i, theta=inputs[i])
    
    for q0, q1 in zip(qubits, qubits[1:]):
        ## IQP circuit可能非常不好实现
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

    ## variational circuit
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9

        # for qubit in range(n_qubits): ## reuploading data encoding
        #     c.ry(qubit, theta=inputs[qubit])
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

    ## last variational layer for constructing Haar unitary
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    ## 计算expectation实际上是可以通过shot进行估计的。那是一种更加实际的做法。
    ## 这里采用rx门来代替z门进行测量，
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([Rx(training_basis[i]), [i]])) for i in range(n_qubits)] 
            # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
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
    # combination = list(combinations(qubits, 2))
    # encoding the qubits as the 1-D Heisenberg model evolution ansatz.
    for qubit in qubits:
        # implemente a Haar random unitary to transforme the |0> into |\psi>
        c.any(qubit, unitary=Haar(qubit))
    for _ in range(T):
        # repeate it for T time slots.
        for i, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
            c.RXX(q0,q1,theta=inputs[i]*(t/T))
            c.RYY(q0,q1,theta=inputs[i]*(t/T))
            c.RZZ(q0,q1,theta=inputs[i]*(t/T))
            ## the number of inputs is still n_qubits - 1 

    ## variational circuit
    for layer in range(n_layers):  ## 每一行的所有参数都表示所有的变分参数 theta0-----theta9
        # Variational layer
        for qubit in qubits:   
            # one_qubit_rotation(c, qubit, weights_var[layer, qubit, :])
            c.rx(qubit, theta=weights_var[layer,qubit, 0])
            c.rz(qubit, theta=weights_var[layer,qubit, 1])

        ## training entangling layer 17个纠缠门
        entangling_layer_trainable(c, qubits, weight_ent[layer, :])

    ## last variational layer for constructing Haar unitary
    for qubit in qubits:
        c.rx(qubit, theta=q_weights_final[qubit, 0])
        c.rz(qubit, theta=q_weights_final[qubit, 1])
    
    ## 计算expectation实际上是可以通过shot进行估计的。那是一种更加实际的做法。
    ## 这里采用rx门来代替z门进行测量，
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n_qubits)]
        # + [K.real(c.expectation([Rx(training_basis[i]), [i]])) for i in range(n_qubits)] 
            # +[K.real(c.expectation([tc.gates.y(), [i]])) for i in range(n_qubits)]
    )

    outputs = K.reshape(outputs, [-1])
    return outputs
    




