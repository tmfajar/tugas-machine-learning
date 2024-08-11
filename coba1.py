import numpy as np 
np.random.seed(213)
#import matplotlib.pyplot as plt
#matplotlib inline


def affine_forward(X, W, b):
 	V = np.dot(X, W) + b
 	return V


def affine_backward(dout, X, W, b):
 	dX = np.dot(dout, W.T)
 	dW = np.dot(X.T , dout)
 	db = np.sum(dout, axis=0, keepdims=True)
 	return dX , dW, db

def sigmoid_forward(V):
 	act = 1/(1+np.exp(-V))
 	return act

def sigmoid_backward(dout, act):
 	dact = act-act**2
 	dout = dout*dact
 	return dout

def train_single_layer(X, y, lr, max_epoch, weights = None, history = None):
	n_data, n_dim = X.shape
	_, n_out = y.shape

	if weights is None:
		W1 = 2 * np.random.random((n_dim, n_out)) - 1
		b1 = np.zeros((1, n_out))
		history = []
	else:
		W1, b1 = weights

	for ep in range(max_epoch):
		#forward pass
		V1 = affine_forward(X, W1, b1)
		y_hat = sigmoid_forward(V1)

		#calculate loss
		E =  y-y_hat
		mse = np.mean(E**2)
		history.append(mse)

		acc = np.sum(y==np.round(y_hat))/n_data
		print('epoch : %i/%i, mse :  %.7f, acc : %.2f' %(ep, max_epoch, mse, acc))

		#backward pass
		dV1 = sigmoid_backward(E, y_hat)
		dx, dW1, db1 = affine_backward(dV1, X, W1, b1)

		#weights update
		W1 = W1 + lr*dW1
		b1 = b1 + lr*db1

	weights = (W1, b1)

	return history, weights


X = np.array([ [0,0], [0,1], [1,0], [1,1] ])
y = np.array([ [0,0,0,1] ]).T


print(X)
print(y)

print('percobaan ke1')
histl, model1 = train_single_layer(X, y, lr=1, max_epoch=10)


def show_graph(history, size =[9,5]):
	plt.rcParams['figure.figsize'] = size
	plt.plot(history)
	plt.xlabel('epoch')
	plt.ylabel('Training MSE')
	plt.title('Training MSE History')
	plt.show()

show_graph(histl)

def test_single_layer(X, weights):
	W1, b1 = weights
	#forward pass
	V1 = affine_forward(X, W1, b1)
	y_hat = sigmoid_forward(V1)

	return np.round(y_hat)

y_pred = test_single_layer(X, model1)
print(y_pred)

print('percobaan ke2')
histl, model1 = train_single_layer(X, y, lr=1, max_epoch=10, weights=model1, history=histl)
show_graph(histl)

y_pred = test_single_layer(X, model1)
print(y_pred)