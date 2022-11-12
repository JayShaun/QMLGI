
import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import struct
import os
# from tensorcircuit import keras
import pickle
from tensorflow import keras
import argparse
from functools import partial
from model import quantum_circuit, quantum_circuit_Noise, quantum_circuit_TB, quantum_circuit_fixing, quantum_circuit_fixing_wout, quantum_circuit_simple
from model import quantum_circuit_O, quantum_IQP, quantum_Heisenberg

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

parser = argparse.ArgumentParser(description='Quantum-Classical NN for SPI')
parser.add_argument('--cofficents-path', metavar='G', default="./cofficients.txt",
                    help='path of label')                                             
# parser.add_argument('--notfixing', type=bool, default=True,
#                     help='not fixing the tuning parameters')
parser.add_argument('--use-schedule', type=bool, default=False,
                    help='use learning rate schedule')    
parser.add_argument('--use-CS', type=bool, default=False,
                    help='use classical shadow')    
parser.add_argument('--QNN-layers', type=int, default=4, metavar='L',
                    help='QNN layers (default: 4)')  
parser.add_argument('--QNN-qubits', type=int, default=16, metavar='Q',
                    help='QNN qubits (default: 16)')                                
parser.add_argument('--learning-rate', type=float, default=0.01, metavar='G',
                    help='lr (default: 0.01)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='minimal batch size (default: 64)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--run-ID', type=int, default=0, metavar='I',
                    help="run-ID")
parser.add_argument('--dim', type=int, default=16, metavar='D',
                    help="dim data")
parser.add_argument('--classes', type=int, default=10, metavar='C',
                    help="number of classes")
parser.add_argument('--encoding', type=str, default="reuploading_simple", metavar='E',
                    help="types of data encodings")


args = parser.parse_args()


np.random.seed(args.seed)
K = tc.set_backend("tensorflow")

cofficients = []
f = open(args.cofficents_path) 
for line in f:
    cofficients.append(float(line.strip()))
    print(cofficients)

args.train_img_path = "./train_dim_{}.npy".format(args.dim)
args.test_img_path = "./test_dim_{}.npy".format(args.dim)
args.train_label_path = "./train-labels-idx1-ubyte"
args.test_label_path = "./test-labels-idx1-ubyte"

if args.dim == 16:
    cofficient = cofficients[0]
elif args.dim == 32:
    cofficient = cofficients[1]
elif args.dim == 64:
    cofficient = cofficients[2]
elif args.dim == 128:
    cofficient = cofficients[3]
else:
    raise ValueError("Unknown dimension of data input.")

# print(args, flush=True)

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_path, label_path, batch_size=args.batch_size, \
        n_classes=args.classes, shuffle=True, use_CS = args.use_CS):
        'Initialization'

        self.batch_size = batch_size
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        self.labels = labels
        self.training_data = np.load(img_path) / cofficient 
        self.dim = self.training_data.shape[1]
        self.list_IDs = range(self.training_data.shape[0])
    
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_CS = use_CS
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.use_CS:
            X = np.empty((self.batch_size, self.dim*2))
            basis = np.random.choice(4, size=(self.batch_size,self.dim), p=[1/4, 1/4, 1/4, 1/4])
        else:
            X = np.empty((self.batch_size, self.dim))

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.use_CS:
                X[i,] = np.reshape(np.stack((self.training_data[ID,], basis[i,])),[-1])
            else:
                X[i,] = self.training_data[ID,]
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# ql = tc.keras.QuantumLayer(quantum_circuit_fixing, [(args.QNN_layers, args.QNN_qubits, 3),\
#     (args.QNN_qubits, 2)], name="qlayer")


class multi_head_ql(tf.Module):
    def __init__(self, N_heads, encodings, name=None):
        super(multi_head_ql, self).__init__(name=name)
        self.qlayers = []
        self.N_heads = N_heads
        with self.name_scope:
            if encodings== "IQP":
                for i in range(N_heads):
                    self.qlayers.append(tc.keras.QuantumLayer(quantum_IQP, [(args.QNN_layers, args.QNN_qubits, 3),\
                        (args.QNN_qubits, 2)], name="qlayer"+str(i)))
            if encodings=="Heisenberg":
                for i in range(N_heads):
                    self.qlayers.append(tc.keras.QuantumLayer(quantum_Heisenberg, [(args.QNN_layers, args.QNN_qubits+1, 3),\
                        (args.QNN_qubits+1, 2)], name="qlayer"+str(i)))
            if encodings=="reuploading_simple":
                for i in range(N_heads):
                    self.qlayers.append(tc.keras.QuantumLayer(quantum_circuit_fixing, [(args.QNN_layers, args.QNN_qubits, 3),\
                        (args.QNN_qubits, 2)], name="qlayer"+str(i)))
    
    @tf.Module.with_name_scope
    def __call__(self, inputs):

        x = tf.split(inputs, self.N_heads, axis=1)
        qlayers_out = []
        for i,layer in enumerate(self.qlayers):
            qlayers_out.append(layer(x[i]))

        cat_heads_out = tf.concat(qlayers_out, axis=1)
        
        return cat_heads_out

def generate_hybrid_model(input_tensor):
    """
    The function is used to generate the hybrid quantum-classical model
    """
    # N_head = int(args.dim/args.QNN_qubits)
    cout = tf.keras.layers.Dense(16, activation='tanh', name="clayer1")(input_tensor)

    quantum_output = multi_head_ql(N_heads=2, encodings=args.encoding)(cout) ## outputshape is B, 16*heads
    
    logits= tf.keras.layers.Dense(10, name="clayer2")(quantum_output)
    ## cumstom bias with basis input
    return logits

# ## data input added with basis input (equal size)
input_tensor = keras.Input(shape=(args.dim,), dtype=tf.float32, name="classicalinput")
logits = generate_hybrid_model(input_tensor)
model = tf.keras.Model(inputs=[input_tensor], outputs=logits)


# dense1_layer_model = tf.keras.Model(inputs=model.input,
#           outputs=model.get_layer('clayer1').output)
# #以这个model的预测值作为输出
# qlayer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("qlayer").output)


def train():

    train_generator = DataGenerator(img_path=args.train_img_path, label_path=args.train_label_path)
    test_generator= DataGenerator(img_path=args.test_img_path, label_path=args.test_label_path)

    print(args, flush=True)
    print(model.summary(), flush=True)
    print("10 classification: \n")
    print("simple quantum circuit hybrid style and dim {}".format(args.dim))

    lr_schedule = args.learning_rate

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.0,
        axis=-1,
        reduction="auto",
        name="categorical_crossentropy",
    ),
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        metrics=['accuracy'],
    )


    checkpoint_filepath = './model_save_{}_qubits_{}_dim{}_encodings_{}_CQC_{epoch}'.format(args.run_ID,args.QNN_qubits, args.dim, args.encoding)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_freq='epoch', period=100)

    history = model.fit_generator(generator=train_generator,
                        epochs=500,
                        verbose=2,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=8, callbacks=[model_checkpoint_callback])

    with open('./trainHistoryDict_{}_layers{}_dim_{}_encodings'.format(args.run_ID, args.QNN_layers, args.dim, args.encoding), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    ## 保存额外的中间变量
    # test_data = np.load(args.test_img_path) / cofficient
    # clayer1_output = dense1_layer_model.predict(test_data)
    # qlayer_output = qlayer_model.predict(test_data)

    # np.savez("./interoutput_ID{}_dim{}_encodings{}".format(args.run_ID, args.dim, args.encoding), clayer1_output = clayer1_output.numpy(), qlayer=qlayer_output.numpy())
    model.save('./model_save_{}_qubits_{}_dim{}_encodings_{}_CQC'.format(args.run_ID,args.QNN_qubits, args.dim, args.encoding))

if __name__ == "__main__":
    train()
