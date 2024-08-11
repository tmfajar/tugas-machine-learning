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