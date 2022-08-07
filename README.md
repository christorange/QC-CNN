![image](https://user-images.githubusercontent.com/85213835/180037547-c1122206-6254-4c9a-96b0-8b0cfae4751d.png)
----

# Quick instruction
This is a Python project simulating a Quantum-classical hybrid convolutional neural network(QC-CNN) for classical image classification. A Chinese version of detailed code analysis can be found [here](https://zhuanlan.zhihu.com/p/516363287).

Directory structure:
```
|—— run.py
|—— app
|     |—— load_data.py
|     |__ train.py
|—— model
|     |—— calssical.py  // a classical CNN with one concolutional layer
|     |—— single_encoding.py  // a quantum-classical hybrid model using single encoding method
|     |—— multi_encoding.py  // a hybrid model using multi encoding method
|     |—— inception.py  // this model contains a quantum-classical hybrid inception module
|     |__ multi_noisy.py  // same model as multi_encoding.py but simulating mixed states
|—— datasets
      |__ // csv datasets files
```
* Run `run.py`.
* Edit console printing outputs in `train.py`.
* Network structures are defined in the files in `model` directory. Chose the model to be used in training by importing it in `run.py`.
* The default datasets files in `datasets` are csv format data of MNIST and FashionMNIST images subsampled to $14 \times 14$ size. Each dataset three types of data, for example `fashion_012` means The data loader for local datasets is `load_data.py` in `app`. 

# Model introduction

A QC-CNN is constructed with both classical neural network layers and quantum circuits. The quantum circuits are also named as *quantum convolutional layers* in some papers. In this project, data is firstly fed into a quantum circuit, the output feature maps are then fed into fully connected layers and then give the classification result. The order of layers and types of classical 


