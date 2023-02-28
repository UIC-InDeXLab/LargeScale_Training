import random
import time
from Regular_NN.nn_layer import Layer
import numpy as np
from time import perf_counter
import tensorflow_datasets as tfds
# from Regular_NN.reg_nn import NeuralNetwork as Regular_NN
from collections import defaultdict


class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_hidden_layers, num_outputs, lr, U, m, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):

        self.LEARNING_RATE = lr
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.U = U  # Max norm
        self.hidden_layers = []
        self.random_neurons = 0
        self.num_hidden_layers = num_hidden_layers
        self.output_layer = Layer(num_outputs, num_hidden, output_layer_bias, U, m)
        self.output_layer.active_set = list(range(num_outputs))
        self.layer_error = np.zeros(num_hidden_layers)


        for i in range(num_hidden_layers):
            prev_nodes = num_inputs
            if i > 0:
                prev_nodes = len(self.hidden_layers[i-1].neurons)
            self.hidden_layer = Layer(num_hidden, prev_nodes, U, m, hidden_=True)
            # curr_nodes = num_hidden

            # if not hidden_layer_weights:
            #     self.weights_from_prevlayer_to_currlayer(prev_nodes, curr_nodes,rand=True)
            #
            # elif hidden_layer_weights:
            #     self.weights_from_prevlayer_to_currlayer(prev_nodes, curr_nodes, hidden_layer_weights[i])
            self.hidden_layers.append(self.hidden_layer)
            self.hidden_layer.prev_layer_node = prev_nodes






    def normalize_output_weights(self):
        if self.output_layer.normalizing_flag:
            for neuron in self.output_layer.neurons:
                neuron.normalize_weight(self.output_layer.MAX_wi / self.U)
            self.output_layer.normalizing_flag = False
            self.output_layer.MAX_wi = self.U



    def feed_forward(self, inputs):
        # self.hidden_layers[0].feed_forward_all(inputs)
        self.hidden_layers[0].active_nodes()
        self.hidden_layers[0].feed_forward(inputs)


        for i in range(1, self.num_hidden_layers):
            # self.hidden_layers[i].feed_forward_all(self.hidden_layers[i-1].get_outputs())
            self.hidden_layers[i].active_nodes()
            self.hidden_layers[i].feed_forward(self.hidden_layers[i-1].get_outputs())

        hidden_layer_outputs = self.hidden_layers[self.num_hidden_layers-1].get_outputs()
        # print(self.hidden_layers[-1].outputs_inactive)
        output = self.output_layer.feed_forward(hidden_layer_outputs)
        return output




    def total_error_test(self, testing_sets, test_size):
        total_error = 0
        classification_error = 0
        dataset = defaultdict(int)


        for t in range(test_size):
            testing_inputs, testing_outputs = testing_sets[t]
            # dataset[np.argwhere(testing_outputs == 1)[0][0]] += 1

            self.clear().feed_forward(testing_inputs)
            # self
            total_err, classification_err = self.output_layer.calculate_error(testing_outputs.flatten())
            total_error += total_err
            classification_error += classification_err

        return total_error, classification_error, self.output_layer.confusion_matrix

    def display(self, error_lsh, error_nolsh):
        print('* Inputs: {}'.format(self.num_inputs))
        print('------'*5)
        for layer in self.hidden_layers:
            print('* Hidden Layer: {}'.format(self.hidden_layers.index(layer)))
            layer.display()
            # print('Collective activations: {}'.format(self.layers_activation[self.hidden_layers.index(layer)][0]))
            # print('Collective activations with LSH: {}'.format(self.layers_activation[self.hidden_layers.index(layer)][1]))
            print('------'*5)
        print('* Output Layer')
        self.output_layer.display()
        print('* Error with LSH : ', error_lsh)
        print('* Error with WTA : ', error_nolsh)
        # print('Total random neurons picked from the hidden layers', self.random_neurons)
        # print('* Average layer error, the difference between collective activations with and without LSH: ')
        # print(self.layer_error)
        print('------'*5)

    def clear(self):
        # print("Here!!!")
        self.layer_error = np.zeros(self.num_hidden_layers)
        self.output_layer.outputs = np.zeros(self.num_outputs)
        self.random_neurons = 0
        for node in self.output_layer.neurons:
            node.clear()
        for layer in self.hidden_layers:
            layer.clear()
        return self

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        # 1. Output neuron deltas
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ = -(tⱼ - yⱼ) * yⱼ * (1 - yⱼ)
            self.output_layer.neurons[o].pd_error_wrt_total_net_input(training_outputs[o][0])
        # 2. Hidden neuron deltas
        self.hidden_layers[(self.num_hidden_layers)-1].backpropagation(self.output_layer)
        for h in range(self.num_hidden_layers-2, 0, -1):
            self.hidden_layers[h].backpropagation(self.hidden_layers[h + 1])
        # 3. Update output neuron weights

        self.output_layer.update_weights_layer(self.LEARNING_RATE)

        # 4. Update hidden neuron weights
        for i in range(self.num_hidden_layers-1, 0, -1):

            self.hidden_layers[i].update_weights_layer(self.LEARNING_RATE)
        return self
        # self.output_layer.display()

    # def weights_from_prevlayer_to_currlayer(self,  prev_layer_nodes, curr_layer_nodes,random_gen, rand= True, hidden_layer_weights=None):
    #     for lc in range(curr_layer_nodes):
    #         if rand:
    #             x = np.random.normal(size=(prev_layer_nodes, 1))
    #             x -= x.mean()
    #             rnd_vec = x / (np.linalg.norm(x))
    #             self.hidden_layer.neurons[lc].weights_lsh = self.hidden_layer.neurons[lc].weights_nolsh = rnd_vec
    #             norm = np.linalg.norm(rnd_vec)
    #             if norm > self.hidden_layer.MAX_wi:
    #                 self.hidden_layer.MAX_wi = norm
    #             # print("Max norm of weight", self.hidden_layer.MAX_xi)
    #                 self.hidden_layer.normalizing_flag = True
    #         else:
    #             self.output_layer.neurons[lc].weights_lsh = self.output_layer.neurons[lc].weights_nolsh = np.array(hidden_layer_weights)


    # def weights_from_hidden_layer_to_output_layer(self, regular_nn_outputlayer, n, output_layer_weights):
    #     # n = self.num_hidden - 1
    #     for o in range(len(self.output_layer.neurons)):
    #         if not output_layer_weights:
    #             x = np.random.normal(size=(n, 1))
    #             x -= x.mean()
    #             rnd_vec = x / (np.linalg.norm(x))
    #             self.output_layer.neurons[o].weights_lsh = self.output_layer.neurons[o].weights_nolsh = rnd_vec
    #             regular_nn_outputlayer.neurons[o].weights = rnd_vec
    #             norm = np.linalg.norm(x)
    #                 # find the max w_i in every layer, for normalizing purposes
    #             if norm > self.output_layer.MAX_wi:
    #                 self.output_layer.MAX_wi = norm
    #                     # print("Max norm of weight", self.output_layer.MAX_xi)
    #                 self.output_layer.normalizing_flag = True
    #         else:
    #             self.output_layer.neurons[o].weights_lsh = self.output_layer.neurons[o].weights_nolsh = np.array(output_layer_weights)
    #             regular_nn_outputlayer.neurons[o].weights = np.array(output_layer_weights)



#
# if __name__ == '__main__':
#
#     ds_train, ds_test = tfds.load(name='mnist', split=["train", "test"], as_supervised=True)
#     training_set = []
#     for i, (image, label) in enumerate(tfds.as_numpy(ds_train)):
#         image = image.flatten()
#         y = np.zeros((10, 1))
#         index = label - 1
#         y[index] = 1
#         training_set.append([image, y])
#
#     testing_set = []
#     for i, (image, label) in enumerate(tfds.as_numpy(ds_test)):
#         image = image.flatten()
#         y = np.zeros((10, 1))
#         index = label - 1
#         y[index] = 1
#         testing_set.append([image, y])
#
#     train_size = 10000  #len(training_set)//1000  # 100
#     test_size = 5   #len(testing_set)//1000 # 100
#     errors = [0]
#     # for num_h_units in [50, 100, 200, 300, 400, 500, 600, 700, 800]:
#     #     for num_h_layers in range(2, 11):
#     nn_regular = Regular_NN(
#         num_inputs=784,
#         num_hidden=50,
#         num_hidden_layers=3,  # 7
#         num_outputs=10,
#         lr=0.01,
#         U=0.83,
#         m=3,
#         hidden_layer_bias=[0.35, 0.2, 0.4, 0.35, 0.2, 0.4, 0.35, 0.2, 0.15, 0.16],
#         output_layer_bias=0.2
#     )
#     nn_lsh = NeuralNetwork(
#         num_inputs=784,
#         num_hidden=50,
#         num_hidden_layers=3,  # 7
#         num_outputs=10,
#         lr=0.01,
#         K_=6,
#         L_=5,
#         U=0.83,
#         m=3,
#         hidden_layer_bias=[0.35, 0.2, 0.4, 0.35, 0.2, 0.4, 0.35, 0.2, 0.15, 0.16],
#         output_layer_bias=0.2
#     )
#     for j in range(50):
#         print(f"Epoch {j} took... ", end='')
#         start_perf = perf_counter()
#         err = 0
#         for i in range(train_size):
#             nn.clear()
#             nn.train(training_set[i][0], training_set[i][1])
#
#         end_perf = perf_counter()
#         print(f"{end_perf - start_perf} seconds")

    # err_lsh, err_regular = nn.total_error_test(training_set, train_size)
    # nn.display(err_lsh / train_size, err_regular / train_size)
    #
    # print("----------------- Testing---------------------")
    # nn.clear()
    # total_err_lsh, total_error_regular = nn.total_error_test(testing_set, test_size)
    # nn.display(total_err_lsh / test_size, total_error_regular / test_size)

