import mnist_loader
import network

training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
net = network.Network([784,10])
net.SGD(training_data,30,10,3.0,test_data=test_data)


Epoch 22: 7553 / 10000
Epoch 23: 7581 / 10000
Epoch 24: 7581 / 10000
Epoch 25: 7569 / 10000
Epoch 26: 7579 / 10000
Epoch 27: 7600 / 10000
Epoch 28: 7587 / 10000
Epoch 29: 7586 / 10000