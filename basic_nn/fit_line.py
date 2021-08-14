import numpy as np
from tensorflow import keras


# A basic function
# we obtain value of Y by mapping X to some function

X = np.array([-1.0,0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0,-1.0,1.0, 3.0, 5.0, 7.0], dtype=float)

# from the above eg we can see that Y can be computed by:=> Y = 2*X -1
# neural networks help us to figure out such rules
# hence we provide both DATA & ANSWERS(LABLES) to program giving us RULES ===> AI/ML


# creating model
# model has 1 layer having 1 neuron & input_shape is just a single value
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# adding hyperparameters
model.compile(optimizer='sgd', loss='mean_squared_error')
#training the model
model.fit(X, Y, epochs=500)

# testing the model
print(model.predict([10]))
print(model.predict([20]))
print(model.predict([-5]))