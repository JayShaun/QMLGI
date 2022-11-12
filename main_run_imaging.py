
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorcircuit as tc
import tensorflow as tf


# from tensorcircuit import keras
import pickle
from tensorflow import keras
import argparse
from functools import partial
from model import quantum_circuit, quantum_circuit_Noise, quantum_circuit_TB, quantum_circuit_fixing, quantum_circuit_fixing_wout
from model import quantum_Heisenberg
from model import quantum_IQP

from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

parser = argparse.ArgumentParser(description='Quantum-Classical NN for SPI')
parser.add_argument('--cofficents-path', metavar='G', default="./cofficients.txt",
                    help='path of label')                                             
# parser.add_argument('--notfixing', type=bool, default=True,
#                     help='not fixing the tuning parameters')
parser.add_argument('--use-schedule', type=bool, default=False,
                    help='use learning rate schedule')    
# parser.add_argument('--use-CS', type=bool, default=False,
#                     help='use classical shadow')    
parser.add_argument('--QNN-layers', type=int, default=3, metavar='L',
                    help='QNN layers (default: 3)')  
parser.add_argument('--QNN-qubits', type=int, default=16, metavar='Q',
                    help='QNN qubits (default: 16)')                                
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='G',
                    help='lr (default: 0.01)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='minimal batch size (default: 256)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--run-ID', type=int, default=0, metavar='I',
                    help="run-ID")
parser.add_argument('--dim', type=int, default=64, metavar='D',
                    help="dim data")
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

args.train_img_path = "./plane_randomP_intensities/train_dim_{}.npy".format(args.dim)
args.test_img_path = "./plane_randomP_intensities/test_dim_{}.npy".format(args.dim)
args.train_label_path = "./plane_crop_train"
args.test_label_path = "./plane_crop_test"

if args.dim == 64:
    cofficient = cofficients[0]
    min_value = 70.74
elif args.dim == 128:
    cofficient = cofficients[1]
    min_value = 60.93
elif args.dim == 256:
    cofficient = cofficients[2]
    min_value = 60.94
elif args.dim == 512:
    cofficient = cofficients[3]
    min_value = 61.02
else:
    raise ValueError("Unknown dimension of data input.")



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_path, label_path, batch_size=args.batch_size, \
        shuffle=True):
        'Initialization'

        self.batch_size = batch_size

        # read label from the folder
        dirs = os.listdir(label_path)
        ## to ensure sort 
        dirs.sort()
        self.num_samples = len(dirs)
        self.true_imgs = []

        for file_dir in dirs:
            img = Image.open(label_path + '/' + file_dir)
            img = np.asarray(img)
            self.true_imgs.append(img)
        
        self.true_imgs = np.stack(self.true_imgs, axis=0)
        self.width = self.true_imgs.shape[2]
        self.height = self.true_imgs.shape[1]

        self.training_data = (np.load(img_path)-min_value) / (cofficient-min_value) ## normlized to 0-pi/2
        self.dim = self.training_data.shape[1]
        self.list_IDs = range(self.num_samples)
    
        # self.n_classes = n_classes
        self.shuffle = shuffle
        
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
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size, self.height, self.width))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.training_data[ID,]
            # Store true images
            y[i] = self.true_imgs[ID, ]/255

        return X, y


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
    # if args.dim==512:
    #     N_head=
    # cout = tf.keras.layers.Dense(256, activation="relu")(input_tensor)
    N_head = int(args.dim/args.QNN_qubits) 
    qlayers_out = multi_head_ql(N_heads=N_head,encodings=args.encoding)(input_tensor)  ## 16*4 = 64
    qlayers_out = tf.keras.layers.Dense(1024, activation='relu')(qlayers_out)
    cnnout = tf.reshape(qlayers_out, (-1, 32, 32, 1))
    cnnout = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same',activation='relu')(cnnout)
    cnnout = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same',activation='relu')(cnnout)
    cnnout = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(cnnout)  
    cnnout = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same',activation='relu')(cnnout)
    cnnout = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(cnnout)  ## 32
    cnnout = tf.keras.layers.Conv2D(8, 3, strides=1, padding='same',activation='relu')(cnnout)
    # cnnout = tf.keras.layers.Conv2D(8, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.Conv2D(8, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(cnnout)  ## 64
    cnnout = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same',activation='relu')(cnnout)
    # cnnout = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same')(cnnout)
    # cnnout = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(cnnout)  ## 64
    reconstructed_img = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')(cnnout) 
    
    ## cumstom bias with basis input
    return tf.squeeze(reconstructed_img, axis=-1)

# ## data input added with basis input (equal size)
input_tensor = keras.Input(shape=(args.dim,), dtype=tf.float32)
reconstructed_img = generate_hybrid_model(input_tensor)
model = tf.keras.Model(inputs=[input_tensor], outputs=reconstructed_img)

print(model.summary(),flush=True)

# dense1_layer_model = tf.keras.Model(inputs=model.input,
#           outputs=model.get_layer('clayer1').output)
#以这个model的预测值作为输出
# qlayer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("qlayer").output)

# model = tf.keras.Sequential([tf.keras.layers.Dense(args.QNN_qubits, activation="tanh"), ql, \
#     tf.keras.layers.Dense(args.classes)])

def train():

    train_generator = DataGenerator(img_path=args.train_img_path, label_path=args.train_label_path)
    test_generator= DataGenerator(img_path=args.test_img_path, label_path=args.test_label_path)

    print(args, flush=True)
    print("image reconstruction: \n")
    print("simple quantum circuit with Upsampling and convolution and dims{}".format(args.dim))

    lr_schedule = args.learning_rate

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"])

    history = model.fit_generator(generator=train_generator,
                        epochs=500,
                        verbose=2,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=8)
    
    with open('./trainHistoryDict_{}_layers{}_dim_{}'.format(args.run_ID, args.QNN_layers, args.dim), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    model.save('./model_save_{}_qubits_{}_dim{}_encodings_{}'.format(args.run_ID,args.QNN_qubits, args.dim, args.encoding))
    

if __name__ == "__main__":
    train()
