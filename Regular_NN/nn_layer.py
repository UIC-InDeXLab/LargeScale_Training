import random
from Regular_NN.nn_neuron import Neuron
import numpy as np


# from sklearn.metrics import jaccard_score

class Layer:

    def __init__(self, num_neurons, prev_layer_node, U, m, bias = None, hidden_=False):

        self.bias = bias if bias else random.random()
        self.neurons = []
        self.hidden = hidden_
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
        self.outputs = np.zeros(num_neurons)
        self.outputs_active = np.zeros(num_neurons)
        self.activations_wta = np.zeros(num_neurons)
        self.n_nodes = num_neurons
        self.active_set = set()
        self.error = 0
        self.n_prev_layer_node = prev_layer_node
        self.MAX_wi = U
        self.U = U
        self.random_nodes = set()
        self.normalizing_flag = False
        if not hidden_:
            self.confusion_matrix = np.zeros((num_neurons, num_neurons))




    def random_node_picking(self):
        # num = ((6 * self.n_nodes) // 100) - len(self.active_set)
        while len(self.active_set) < int(0.05 * self.n_nodes):

            random_node = random.choice(range(0, self.n_nodes-1))
            self.random_nodes.add(random_node)
            self.active_set.add(random_node)

    def active_nodes(self):
        # arr = self.activations_wta.argsort()[::-1][:50]
        self.active_set = set(range(self.n_nodes))

    def get_outputs(self):
        return self.outputs_active

    def feed_forward(self, input):
        for node in self.active_set:
            self.outputs_active[node] = self.neurons[node].neuron_output(input)
        # self.display()
                # self.neurons[node].clear()
                # self.outputs_active[node] = self.outputs[node]

    # def feed_forward_training(self, input):
    #     for node in self.active_set_training:
    #         self.outputs_active[node] = self.neurons[node].neuron_output_training(input)

    def feed_forward_all(self, input):
        # print(self.n_nodes)
        for v in range(self.n_nodes):
            self.neurons[v].total_net_input(input)
            self.activations_wta[v] = self.neurons[v].activation_wta

    def backpropagation(self, next_layer):
        active_set = self.active_set
        active_set_next_layer = next_layer.active_set
        for curr_node in active_set:
            # the derivative of the error with respect to the output of each hidden layer neuron j
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wij
            d_error_wrt_hidden_neuron_output = 0
            for neighbor_node in active_set_next_layer:
                d_error_wrt_hidden_neuron_output += next_layer.neurons[neighbor_node].delta * \
                                                    next_layer.neurons[neighbor_node].weights[curr_node, 0]
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            self.neurons[curr_node].delta = d_error_wrt_hidden_neuron_output * self.neurons[curr_node].pd_total_net_input_wrt_input()

    def update_weights_layer(self, learning_rate):
        active_set = self.active_set
        for h in active_set:
            self.neurons[h].update_weights(learning_rate)


    def calculate_error(self, target_output):
        classification_accuracy = 0
        max_idx = np.argmax(self.outputs_active)
        output = np.zeros(self.n_nodes)
        output[max_idx] = 1
        if target_output[max_idx] == 1:
            classification_accuracy = 1
            self.confusion_matrix[max_idx][max_idx] += 1
        else:

            i = np.argwhere(target_output == 1)[0][0]
            self.confusion_matrix[i][max_idx] += 1

        return np.sum(0.5 * np.power(target_output - self.outputs_active, 2)), classification_accuracy




    def clear(self):
        self.outputs = np.zeros(self.n_nodes)
        self.outputs_active = np.zeros(self.n_nodes)
        self.error = 0

        for node in self.neurons:
            node.clear()
        self.active_set = set()
        self.random_nodes = set()


    def calculate_layer_error(self, output_actives, output_inactives):
        prev_error = self.error
        self.error = np.sum(output_inactives - output_actives)
        return self.error + prev_error

    def display(self):

        if self.hidden:
            print('% of Active Neurons Regular :', (len(self.active_set) * 100) / self.n_nodes)
            print('Collective Output Regular :', np.sum(self.outputs_active))
            # self.neurons[10].display()


