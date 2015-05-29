# kaggle-digit-recognizer

[Kaggle digit recognition](https://www.kaggle.com/c/digit-recognizer) using a neural net in [Theano](https://github.com/Theano/Theano).

Code based around [theano-nn](https://github.com/cgyulay/theano-nn).

----
####Topology
The final network consisted of two fully connected hidden layers and a softmax output layer and was regularized using [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf). Each hidden layer contained 1000 [rectified linear units](http://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29). Training was done on one NVIDIA GeForce GT 750M 2048 MB GPU for ~2 mins which produced a final test accuracy on Kaggle of just over 98%.
