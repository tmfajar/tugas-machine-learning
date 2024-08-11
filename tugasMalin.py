import numpy as np
np.random.seed

def affine_forward(X, W, b):
    V = np.dot(X, W) + b
    return V

def affine_backward(dout, X, W, b):
    dX = np.dot(dout, W.T)
    dW = np.dot(X.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)
    return dX, dW, db

def sigmoid_forward(V):
    act = 1/(1+np.exp(-V))
    return act

def sigmoid_backward(dout, act):
    dact = act-act**2
    dout = dout*dact
    return dout

def train_single_layer(X, y, lr, max_epoch, weights=None, history=None):
    n_data, n_dim = X.shape
    _, n_out = y.shape

    if weights is None :
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
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)

        acc = np.sum(y==np.round(y_hat))/n_data
        print('epoch: %i/%i, mse: %.7f, acc: %.2f' % (ep, max_epoch, mse, acc))

        #backward pass 
        dV1 = sigmoid_backward(E, y_hat)
        dX, dW1, db1 = affine_backward(dV1, X, W1, b1)

        #weights update
        W1 = W1 + lr*dW1
        bl = b1 + lr*db1

    weights = (W1, b1)

    return history, weights

X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1], ])
y = np.array([ [0, 0, 0, 1] ]).T

print(X)
print(y)

print('coba')
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

print('coba 2')
histl, model1 = train_single_layer(X, y, lr=1, max_epoch=10, weights=model1, history=histl)
show_graph(histl)

y_pred = test_single_layer(X, model1)
print(y_pred)



def train_two_layers(X, y, n_hidden, lr, max_epoch, weights=None, history=None, print_every=10):
    n_data, n_dim = X.shape
    _, n_out = y.shape

    if (weights is None):
        W1 = 2 * np.random.random((n_dim, n_hidden)) - 1
        b1 = np.zeros((1, n_hidden))
        W2 = 2 * np.random.random((n_hidden, n_out)) - 1
        b2 = np.zeros((1, n_out))
        history = []

    else:
        W1, b1, W2, b2 = weights

    for ep in range(max_epoch):
        # Forward Pass
        V1 = affine_forward(X, W1, b1)
        A1 = sigmoid_forward(V1)
        V2 = affine_forward(A1, W2, b2)
        y_hat = sigmoid_forward(V2)

        # Calculate Loss
        E = y-y_hat
        mse = np.mean(E**2)
        history.append(mse)

        if (ep % print_every == 0):      
            acc = np.sum(y==np.round(y_hat))/n_data
            print('epoch : %i/%i, mse : %.7f, acc : %.2f' % (ep, max_epoch, mse, acc))

        # Backward Pass
        dV2 = sigmoid_backward(E, y_hat)
        dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
        dV1 = sigmoid_backward(dA1, A1)
        dX, dW1, db1 = affine_backward(dV1, X, W1, b1)

        # Weight Update
        W1 = W1 + lr*dW1
        b1 = b1 + lr*db1
        W2 = W2 + lr*dW2
        b2 = b2 + lr*db2

    weights = (W1, b1, W2, b2)

    return history, weights

def test_two_layers(X, weights):
    W1, b1, W2, b2 = weights

    # forward pass
    V1 = affine_forward(X, W1, b1)
    A1 = sigmoid_forward(V1)
    V2 = affine_forward(A1, W2, b2)
    y_hat = sigmoid_forward(V2)

    return np.round(y_hat)

y = np.array([ [0, 1, 1, 0] ]).T    
print(y)  

print('coba3')
hist2, model2 = train_two_layers(X, y, n_hidden=4, lr=1, max_epoch=100, print_every=10)
show_graph(hist2)

y_pred = test_two_layers(X, model2)
print(y_pred)

print('coba4')
hist2, model2 = train_two_layers(X, y, n_hidden=4, lr=1, max_epoch=300, print_every=10, weights=model2, history=hist2)
show_graph(hist2)

y_pred = test_two_layers(X, model2)
print(y_pred)


print('coba5')
hist3, model3 = train_two_layers(X, y, n_hidden=4, lr=0.1, max_epoch=400, print_every=50)
show_graph(hist3)

y_pred = test_two_layers(X, model3)
print(y_pred)


print('coba6')
hist2, model2 = train_two_layers(X, y, n_hidden=4, lr=0.1, max_epoch=3000, print_every=100, weights=model3, history=hist3)
show_graph(hist3)

y_pred = test_two_layers(X, model3)
print(y_pred)


from sklearn.datasets import make_classification 

COLORS = ['red', 'blue']
DIM = 20
INFO = 10
CLASS = 2
NDATA = 600

Xb, yb1 = make_classification(n_samples=NDATA, n_classes=CLASS, n_features=DIM, n_informative=INFO, n_clusters_per_class=4, flip_y=0.2, random_state=33)
yb = yb1.reshape((-1, 1))



from mpl_toolkits.mplot3d import Axes3D

# show features
ft = [0, 1, 2]

fig = plt.figure(figsize=(10, 6), dpi=100)
ax = Axes3D(fig)
ax.scatter( Xb[yb1==0, ft[0]], Xb[yb1==0, ft[1]], Xb[yb1==0, ft[2]], c=COLORS[0], marker='s' )
ax.scatter( Xb[yb1==1, ft[0]], Xb[yb1==1, ft[1]], Xb[yb1==1, ft[2]], c=COLORS[1], marker='o' )
plt.show()

print('coba7')
hist4, model4 = train_two_layers(Xb, y, n_hidden=4, lr=1, max_epoch=500, print_every=25)
show_graph(hist4)

print('coba8')
hist5, model5 = train_two_layers(Xb, y, n_hidden=4, lr=0.01, max_epoch=500, print_every=25)
show_graph(hist5) 


print('coba9')
hist6, model6 = train_two_layers(Xb, yb, n_hidden=20, lr=0.01, max_epoch=500, print_every=25)
show_graph(hist6) 


def show_n_graph(histories, names, size=[9, 5]):
    plt.rcParams['figure.figsize'] = size
    for i in range(len(histories)) :
        plt.plot(histories[i], label=names[i])
    plt.xlabel('epoch')
    plt.ylabel('tarining mse')
    plt.title('Training MSE History')
    plt.legend() 
    plt.show()   

print('coba10')
show_n_graph([hist5, hist6], ['4 hidden', '20 hidden'])

W1, b1, W2, b2 = model6

print('size data train = ', Xb.shape, ': ', Xb.nbytes, 'byte')
print('size weight W1  = ', W1.shape, ': ', W1.nbytes, 'byte')
print('size weight b1  = ', b1.shape, ': ', b1.nbytes, 'byte')
print('size weight W2  = ', W2.shape, ': ', W2.nbytes, 'byte')
print('size weight b2  = ', b2.shape, ': ', b2.nbytes, 'byte')


V1 = affine_forward(X, W1, b1)
A1 = sigmoid_forward(V1)
V2 = affine_forward(A1, W2, b2)
y_hat = sigmoid_forward(V2)

print('size weight A1     = ', A1.shape, ': ', A1.nbytes, 'byte')
print('size weight y_hat  = ', y_hat.shape, ': ', y_hat.nbytes, 'byte')


E = y-y_hat

dV2 = sigmoid_backward(E, y_hat)
dA1, dW2, db2 = affine_backward(dV2, A1, W2, b2)
dV1 = sigmoid_backward(dA1, A1)
dX, dW1, db1 = affine_backward(dV1, X, W1, b1)

print('size loss matrix = ', E.shape, ': ', E.nbytes, 'byte')
print('size gradient A1  = ', dA1.shape, ': ', dA1.nbytes, 'byte')
print('size gradient W1  = ', dW1.shape, ': ', dW1.nbytes, 'byte')
print('size gradient b1  = ', db1.shape, ': ', db1.nbytes, 'byte')
print('size gradient W2  = ', dW2.shape, ': ', dW2.nbytes, 'byte')
print('size gradient b2  = ', db2.shape, ': ', db2.nbytes, 'byte')
