import math
import numpy as np

class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.activation_wta = 0
        self.delta = 0
        self.inputs = []
        self.output = 0


    def neuron_output(self, inputs):

        self.inputs = np.array(inputs)
        self.output = self.net_output(self.total_net_input(inputs))
        return self.output

    def total_net_input(self, inputs):
        input = np.array(inputs).transpose()
        # print('weight', len(self.weights))
        # print('input', input.shape)
        total = np.matmul(input, self.weights)
        self.activation_wta = total + self.bias

        return total + self.bias

    # Apply the logistic function to calculate the output of the neuron
    def net_output(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def normalize_weight(self, normalizer):
        weight = self.weights
        self.weights = weight / normalizer
    # delta = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def pd_error_wrt_total_net_input(self, target_output):

        self.delta = round((self.pd_error_wrt_output(target_output) * self.pd_total_net_input_wrt_input()), 4)




    # The error for each neuron in the output using Mean Square Error method:


    # def calculate_error_testing(self, target_output, lsh_flag):
    #     if lsh_flag:
    #         return 0.5 * (target_output - self.output_lsh) ** 2
    #     return 0.5 * (target_output - self.output_nolsh) ** 2

    # The partial derivate of the error with respect to actual output:
    # = 2 * 0.5 * (target output - actual output) ^ (1) * -1 = -(target output - actual output)
    # the actual output of the output neuron is yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def pd_error_wrt_output(self, target_output):

        return -(target_output - self.output)

    # using logistic function to calculate the neuron's output:
    # yⱼ = 1 / (1 + e^(-zⱼ))
    # The derivative of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def pd_total_net_input_wrt_input(self):

        return self.output * (1 - self.output)
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zj = netj = x₁w₁ + x₂w₂ ...
    # The partial derivative of the total net input with respective to a given weight is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def pd_total_net_input_wrt_weight(self, idx):
        return self.inputs[idx]

    def update_weights(self, LEARNING_RATE):
        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
        # pd_error_wrt_weight = self.delta * self.pd_total_net_input_wrt_weight()
        # # Δw = α * ∂Eⱼ/∂wᵢ
        # self.weights -=  LEARNING_RATE * pd_error_wrt_weight
        # Δw = α * ∂Eⱼ/∂wᵢ

        for w_ij in range(len(self.weights)):
            # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
            pd_error_wrt_weight = self.delta * self.pd_total_net_input_wrt_weight(w_ij)
            self.weights[w_ij] -= LEARNING_RATE * pd_error_wrt_weight


    def clear(self):
        self.output = 0
        self.inputs = []
        self.delta = 0
        self.activation_wta = 0

    # def display(self):
    #     print('Weights:', self.weights)
        # print('  Output:', self.output)

