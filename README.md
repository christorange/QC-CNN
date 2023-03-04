
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
|     |—— multi_encoding.py  // a hybrid model using multiple encoding method
|     |—— inception.py  // this model contains a quantum-classical hybrid inception module
|     |__ multi_noisy.py  // same model as multi_encoding.py but simulating mixed states
|—— datasets
      |__ // csv datasets files
```
* Run `run.py`.
* Edit console printing outputs in `train.py`.
* Network structures are defined in the files in `model` directory. Select the model to be used in classification by importing it in `run.py`. The default model is a classical CNN.
* The default datasets files in `datasets` are csv format data of MNIST and FashionMNIST images subsampled to $14 \times 14$ size. Each dataset contains 3 types of images, 400 in each type, for example `fashion_012_1200.csv` means totally 1200 images with type 0,1,2 in FashionMNIST. The data loader for local datasets is `load_data.py` in `app`. 

# Dependency (IMPORTANT❗️)
Due to the upgration of PennyLane, models using multi-encoding method (multi_encoding.py & multi_noisy.py) cannot run under latest version of PennyLane. To run these two models, please use PennyLane v0.23.0, and downgrade `autoray` to 0.2.5:
```
pip uninstall autoray
pip install autoray==0.2.5
```
If you are using MacOS, install jax with  `conda install jax -c conda-forge`.

This problem is due to this line of code:
```
exec('qml.{}({}, wires = {})'.format(encoding_gates[i], inputs[qub * var_per_qubit + i], qub))
```
For now I'm not fixing this problem since I'm not actively working on PennyLane, if you are familiar with latest version of PennyLane **you are very welcomed to commit to this project :)**

# Model introduction

A QC-CNN is constructed with both classical neural network layers and quantum circuits. The quantum circuits are also named as *quantum convolution kernels* in some papers. In this project, data is firstly fed into a quantum circuit, the output feature maps are then fed into fully connected layers and then give the classification result. You can freely build your own model and put it in `model` without changing other parts of the project.

![image](https://user-images.githubusercontent.com/85213835/180037547-c1122206-6254-4c9a-96b0-8b0cfae4751d.png)

# Quantum convolution kernel

A typical quantum convolution kernel contains an encoding module, to encode classical data into quantum states, and a trainable entangling module, to extract features from data. The following picture shows the quantum convolution kernel in `single_encoding.py`. It uses single encoding method, encoding one classical data $x_i$ into one qubit.

![image](https://user-images.githubusercontent.com/85213835/183294114-0c5b538d-c4ef-44f7-a621-5e12197511cb.png)

When encoding multiple data into one qubit, it is called mutiple encoding method, the multiple encoding module in `multi_encoding.py` is:

![image](https://user-images.githubusercontent.com/85213835/183294874-27980db8-f7b2-4d9c-8488-6fa716bf34fc.png)

Note that the kernel size is correspondingly changed when using different encoding methods.

# Quantum-classical hybrid Inception module

Inception module is a sturcture with parallel convolution kernels proposed in GoogLeNet. What if building one with quantum convolution kernels? The hybrid Inception module in `inception.py` is:

![image](https://user-images.githubusercontent.com/85213835/183295269-93a3176d-8517-4c69-b0ea-ae004b2717e4.png)

Among all the models in `model`, this one has the best performance :P.




