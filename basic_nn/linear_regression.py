import numpy as np


def feedforward(x, weights):
    '''
    guess the values
    '''
    return np.dot(x, weights)


def mean_square_error(yhat, y):
    '''
    loss function
    '''
    return np.sum(np.square(np.subtract(yhat,y))) / y.size


def backpropagation(yhat, y, learning_rate, weights):
    error = np.subtract(yhat,y)
    gradients = np.dot(error, x)
    adjusted_weights = weights - learning_rate* gradients
    return adjusted_weights




''' price of 1 house bedroom is 100, 2 house bedroom is 150 & so on.
    Train a model to predict house price given no. of bedrooms '''

''' Find the best fitting line through the data points via Gradient Descent '''

x = np.array([1,2,3,4,5,6,11,12,13,14,15,16,19], dtype=float)
y = np.array([100,150,200,250,300,350,600,650,700,750,800,850,1000], dtype=float)

# add column of ones for y-intercept
x = np.concatenate((np.ones((x.size,1), dtype=float), x.reshape(x.size,1)), axis=1)

# Our initial guess
weights = np.random.rand(x.shape[-1])
print('initial weights', weights)


# Doing 1000 iterations
epochs = 10000
learning_rate = 0.001

for epoch in range(epochs):
    yhat = feedforward(x, weights)
    # Calculate and print the error
    if epoch % 1000 == True:
        loss = mean_square_error(yhat, y)
        print(f'Epoch {epoch} --> loss: {loss}')
    weights = backpropagation(yhat, y, learning_rate, weights)


print(f'Best estimates: {weights}')
test_x = np.array([1, 7])
print(f'{test_x} bedroom house price {np.dot(test_x, weights)}')
test_x = np.array([1, 8])
print(f'{test_x} bedroom house price {np.dot(test_x, weights)}')
