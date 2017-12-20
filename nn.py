import numpy as np
import math
import logging

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger("Neural Network")

# The neural network weight and biases
layers = []

# Activation functions
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def rectifier(x):
    return np.maximum(x, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
# End activation functions

# Create neural network. First integer in the list is the input layer. Last number is the output layer
def init_neural_network(layers_size):

    weights = []
    biases = []

    logger.info("Starting network initialization: {}\n".format(layers_size))

    # Create all weights
    current_position = 0
    for layer_size in layers_size[:-1]:

        # Initialize random weights and biases between -1 and 1
        layer_weights = (np.random.random((layers_size[current_position+1], layer_size)) * 2) - 1
        layer_biases = (np.random.random((layers_size[current_position+1]))  * 2) - 1
        weights.append(layer_weights)
        biases.append(layer_biases)

        logger.debug("Weights between Layer {} ({} nodes) and Layer {} ({} nodes):\n{}\n".format(
            current_position,
            layer_size,
            current_position + 1,
            layers_size[current_position+1],
            layer_weights))

        logger.debug("Biases between Layer {} ({} nodes) and Layer {} ({} nodes):\n{}\n".format(
            current_position,
            layer_size,
            current_position + 1,
            layers_size[current_position+1],
            layer_biases))

        current_position += 1

    return (weights, biases)

# Layers and input vector
def forward_pass(layers, inputs):

    # Check if the input vector is the same size as the input layer
    if (len(inputs) != layers[0][0].shape[1]):
        logger.error("Input does not match neural network")
        return

    logger.debug("-----------------------------------------\n Starting forward pass \n")
    logger.debug("Input vector: {}".format(inputs))

    weights = layers[0]
    biases = layers[1]

    activation_function = np.vectorize(rectifier)

    result_vector = inputs
    for layer_weights, layer_biases in zip(weights[:-1], biases[:-1]):
        result_vector = activation_function(np.dot(layer_weights, result_vector) + layer_biases)

        logger.debug("Hidden layer vector: {}".format(result_vector))

    # Output layer with softmax activation function
    result_vector = softmax(np.dot(weights[-1], result_vector) + biases[-1])
    logger.debug("Output vector: {}".format(result_vector))

    return result_vector

def main():
    xor =  np.array([[1, 1], [0, 1], [1, 0], [0, 0]])

    layers = init_neural_network([2, 2, 2, 1])

    # TODO: Learn

    # Check if the Neural Network passes the XOR test
    test_succes = True

    logger.info("Starting neural network XOR test")
    # Start predicting the XOR output values
    for input in xor:
        output_vector = forward_pass(layers, input)

        if output_vector == np.array([bool(input[0]) ^ bool(input[1])]):
            logger.debug("Wrong prediction.\n")
            test_succes = False
        else:
            logger.debug("Correct prediction.\n")

    logger.info("XOR test succes: {}".format(test_succes))

if __name__ == "__main__":
    main()