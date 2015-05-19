import os
import theano
import time
import cPickle

import numpy as np
import pandas as pd
import theano.tensor as T

from theano import config

config.mode='FAST_COMPILE'
config.floatX='float32'
data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')


# Activations

def linear(x):
  return x

def sigmoid(x):
  # T.nnet.sigmoid
  return 1.0 / (1.0 + T.exp(-x))

def tanh(x):
  return (T.exp(x) - T.exp(-x)) / (T.exp(x) + T.exp(-x))

def relu(x):
  return T.maximum(x, 0.0)

def relu_leaky(x, alpha=3.0):
  return T.maximum(x, x * (1.0 / alpha))


# Costs

def ce_multiclass(py_x, y):
  # Locations 1,2,3,...,n by y
  # return T.nnet.categorical_crossentropy(py_x, y)
  return -T.mean(T.log(py_x)[T.arange(y.shape[0]), y])

def ce_binary(py_x, y):
  # T.nnet.binary_crossentropy
  return T.mean(T.nnet.binary_crossentropy(py_x, y))

def mse(pred, y):
  return T.mean((pred - y) ** 2)


# Network

class ConnectedLayer():

  def __init__(self, input, n_in, n_out, rng, activation=relu, p_dropout=0.0, w=None, b=None, input_dropout=None):
    self.input = input
    self.n_in = n_in
    self.n_out = n_out
    self.activation = activation
    self.p_dropout = p_dropout

    if w == None:
      w = init_weights((n_in, n_out))

    if b == None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    if input_dropout != None:
      self.input_dropout = dropout(input_dropout, p_dropout, rng)
      self.output_dropout = activation(T.dot(input_dropout, self.w) + self.b)

    self.output = activation((1 - self.p_dropout) * T.dot(input, self.w) + self.b)
    self.y_pred = T.argmax(self.output, axis=1)
    self.params = [self.w, self.b]


class SoftmaxLayer():

  def __init__(self, input, n_in, n_out, rng=np.random.RandomState(1234), p_dropout=0.0, w=None, b=None, input_dropout=None):
    self.input = input
    self.n_in = n_in
    self.n_out = n_out
    self.p_dropout = p_dropout

    if w == None:
      w = init_weights((n_in, n_out))

    if b == None:
      b = init_weights((n_out,))

    self.w = w
    self.b = b

    if input_dropout != None:
      self.input_dropout = dropout(input_dropout, p_dropout, rng)
      self.output_dropout = softmax(T.dot(input_dropout, self.w) + self.b)

    self.output = softmax((1 - self.p_dropout) * T.dot(self.input, self.w) + self.b)
    self.y_pred = T.argmax(self.output, axis=1)
    self.params = [self.w, self.b]


class NN():

  def __init__(self, lr=0.5, batch_size=100, n_hidden=2000, n_epochs=100, test=False, regularization=None, L1_reg=0.01, L2_reg=0.0001):
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.regularization = regularization

    # Load data and unpack
    if not test:
      datasets = load_training_data()
      train_set_x, train_set_y = datasets[0]
      test_set_x, test_set_y = datasets[1]

      # Initialize blank weights and biases
      w_b = [[None, None], [None, None], [None, None]]

      # Number of minibatches
      n_train_batches = len(train_set_x.get_value(borrow=True)) / batch_size
      n_test_batches = len(test_set_x.get_value(borrow=True)) / batch_size

      # Number of target classes
      classes = set()
      for item in train_set_y.eval():
        classes.add(item)
      n_output_classes = len(classes)

    else:
      test_set_x = load_test_data()

      # Initialize saved weights and biases
      w_b = load_saved_weights()

      # Number of target classes
      n_output_classes = 10

    print('Building the model...')

    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)

    # Set up dropout regularization if necessary
    p_dropout_input = 0.0
    p_dropout_hidden = 0.0
    if self.regularization == 'dropout':
      p_dropout_input = 0.2
      p_dropout_hidden = 0.5

    h1 = ConnectedLayer(
      input=x,
      input_dropout=x,
      n_in=28 * 28,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=p_dropout_input,
      w=w_b[0][0],
      b=w_b[0][1]
    )

    h2 = ConnectedLayer(
      input=h1.output,
      input_dropout=h1.output_dropout,
      n_in=n_hidden,
      n_out=n_hidden,
      rng=rng,
      activation=relu,
      p_dropout=p_dropout_hidden,
      w=w_b[1][0],
      b=w_b[1][1]
    )

    softmax = SoftmaxLayer(
      input=h2.output,
      input_dropout=h2.output_dropout,
      n_in=n_hidden,
      n_out=n_output_classes,
      rng=rng,
      p_dropout=p_dropout_hidden,
      w=w_b[2][0],
      b=w_b[2][1]
    )

    # It looks ridiculous but just flattens all layer params into one list
    self.layers = [h1, h2, softmax]
    params = [layer.params for layer in self.layers]
    self.params = [param for subparams in params for param in subparams]

    # L1 and L2 regularizations
    L1 = sum([l.w.sum() for l in self.layers])
    L2 = sum([(l.w**2).sum() for l in self.layers])

    # Construct model
    output_layer = self.layers[-1]
    py_x = output_layer.output # softmax output = P(y|x)
    py_x_dropout = output_layer.output_dropout

    y_pred = output_layer.y_pred
    accuracy = T.mean(T.eq(y_pred, y))

    if self.regularization == 'dropout':
      print('Using dropout regularization...')
      cost = ce_multiclass(py_x_dropout, y)
    elif self.regularization == 'L1':
      print('Using L1 regularization...')
      cost = ce_multiclass(py_x + L1_reg * L1, y)
    elif self.regularization == 'L2':
      print('Using L2 regularization...')
      cost = ce_multiclass(py_x + L2_reg * L2, y)
    else:
      print('No regularization specified, I hope you know what you\'re doing ;)')
      cost = ce_multiclass(py_x, y)

    updates = rmsprop(cost, self.params, self.lr)

    print('Compiling functions...')

    # Use givens to specify numeric minibatch from symbolic x and y
    index = T.lscalar()

    if not test:
      train = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
          x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
          y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        }
      )

      test_accuracy = theano.function(
        inputs=[index],
        outputs=accuracy,
        givens={
          x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
          y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        }
      )

      print('Beginning training!')

      for epoch in xrange(n_epochs):
        start_time = time.clock()

        for start in xrange(n_train_batches):
          cost = train(start)

        total_test_accuracy = np.mean([test_accuracy(i) for i in xrange(n_test_batches)])

        end_time = time.clock()
        print('\nEpoch %d of %d took %.1fs\nTest accuracy: %.2f%%' % ((epoch + 1), n_epochs, (end_time - start_time), (total_test_accuracy * 100)))

        # Reduce learning rate periodocially
        if (epoch + 1) % 25 == 0:
          self.lr /= 10
          print('Reducing learning rate to %f.' % self.lr)

        # Save weights for competition testing
        if (epoch + 1) % n_epochs == 0:
          save_weights(self.layers)

    else:
      predict = theano.function(
        inputs=[x],
        outputs=y_pred
      )

      print('Beginning testing!')

      predictions = predict(test_set_x.eval())

      save_path = os.path.join(data_path, 'predictions.csv')
      print('Saving predictions to %s.' % save_path)

      f = open(save_path, 'w')
      f.write('ImageId,Label\n')

      count = 0
      for pred in predictions:
        count += 1
        f.write('{0},{1}\n'.format(count, pred))

      f.close()


# Loading + saving data

def load_training_data():
  def shared_dataset(x, y, borrow=True):
    # store dataset in theano shared variables to utilize gpu
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
    
    return shared_x, T.cast(shared_y, 'int32')

  dataset = os.path.join(data_path, 'train.csv')
  images = pd.read_csv(dataset, sep=',')

  values = images.values
  y = values[:, 0] # Labels in first col
  x = values[:, 1:] # Pixel data in following 784 cols

  # Divide into training and test data (separate from competition provided test data)
  train_set_x = x[:35000, :]
  train_set_y = y[:35000]

  test_set_x = x[35000:42000, :]
  test_set_y = y[35000:42000]
  
  return [shared_dataset(train_set_x, train_set_y), shared_dataset(test_set_x, test_set_y)]

def load_test_data():
  dataset = os.path.join(data_path, 'test.csv')
  images = pd.read_csv(dataset, sep=',')

  x = images.values
  return theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

def save_weights(layers):
  save_path = os.path.join(data_path, 'weights.pkl')
  print('Training complete. Saving weights to %s' % save_path)

  w_b = []
  for layer in layers:
    w_b.append((layer.w, layer.b))

  cPickle.dump(w_b, open(save_path, 'wb'))

def load_saved_weights():
  weights_path = os.path.join(data_path, 'weights.pkl')
  return cPickle.load(open(weights_path, 'rb'))


# Gradient

# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf, slide 28
def rmsprop(cost, params, lr, rho=0.9, epsilon=1e-6):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    mean_square = theano.shared(p.get_value() * 0.0)

    mean_square_new = rho * mean_square + (1 - rho) * g ** 2
    g_scale = T.sqrt(mean_square_new + epsilon)
    g = g / g_scale

    updates.append((mean_square, mean_square_new))
    updates.append((p, p - lr * g))
  return updates


# Special layers

def softmax(x):
  # T.nnet.softmax
  e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x')) # Make column out of 1d vector
  return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(x, p=0.0, rng=np.random.RandomState(1234)):
  # Binomial distribution
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(4321))

  if p > 0:
    p = 1 - p # 1 - p because p = probability of dropping

    # Create distribution with p successes over x.shape experiments
    x *= srng.binomial(x.shape, p=p, dtype=theano.config.floatX)
    x /= p
  return x


# Utilities

def floatX(x):
  return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
  return theano.shared(floatX(np.random.randn(*shape) * 0.01))


# Run

if __name__ == '__main__':
  lr = 0.0001
  regularization = 'dropout'
  n_epochs = 30

  # Train
  NN(lr=lr, regularization=regularization, n_epochs=n_epochs)

  # Run
  NN(lr=lr, regularization=regularization, test=True)

