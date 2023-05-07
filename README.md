## This code is used to generate the results of the paper "Practical advantage of quantum machine learning in ghost imaging."


The trained model in QML is large and we cannot upload here. One can run the code directly to generate the results presented in the paper. 

The calssical machine learning model is better to be executed in GPU device for fast training. The QML model have more specifications. One can only use CPU to train the QML model when given the number of qubits to be 8. For GPU device, one can set the number of qubits to be 16. We note that the case of 8 qubits has faster training speed when given mulpli CPU cores such as 40 or 50. 

CML code implementation can refer to the folder ./code/CML/. QML code implementation can refer to the folder ./code/QML/
We have provided detailed explanations of our code. main_run_idf.py and main_run_imaging.py are two files can directly be executed to perform indentification and imaging task. 
