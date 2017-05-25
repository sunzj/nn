#! /usr/bin/env python

import numpy as np
import random

class Neuron:

    def __init__(self,bias):
        self.bias         = bias
        self.weights      = []
        self.d_weights    = []
        self.d_bias       = []
        self.back_output  = 0.0
        self.back_weights = []
        self.nums_inputs  = 0
        self.active_type = 'sigmoid'

    def init_weights(self,nums_inputs):
	self.nums_inputs = nums_inputs
        self.weights     = [0] * nums_inputs
        for i in range(nums_inputs):
            self.weights[i] = random.random()

    def set_mini_batch(self,mini_batch):
        d_weight       = [0.0] * mini_batch
        self.d_bias    = [0.0] * mini_batch
        for i in range(self.nums_inputs):
            self.d_weights.append(d_weight)

    def calculate_total_input(self):
        total = 0.0   
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias   

    def activate(self,total_input):
        if self.active_type == 'sigmoid':
            return 1 / (1 + np.exp(-total_input))
        if self.active_type == 'relu':
            return max(0,total_input) 
			
    def d_activate(self):
        if self.active_type == 'sigmoid':
            return self.output * (1 - self.output)
        if self.active_type == 'relu':
            if self.output > 0:
               return 1
            else:
               return 0

    def calculate_output(self,inputs):
        self.inputs  = inputs
        self.output  = self.activate(self.calculate_total_input())
        return self.output

    def calculate_loss(self,target_output):
        loss = 0.0
        loss += target_output * np.log(self.output) + (1 - targe_output) * np.log(1-self.output)
        return loss * (-1)

    def calculate_dloss_dy(self,target_output):
        return (self.output - target_output) / (self.output * (1 - self.output))

    def calculate_dy_dz(self):
	return self.d_activate()
        
    def calculate_dz_dw(self,index):
        return self.inputs[index]
        
    def calculate_dz_db(self):
        return 1

    def flash_back_weights(self,index,input_neurons):
        
        if len(self.back_weights) != len(input_neurons):
            self.back_weights = [0] * len(input_neurons)
        for i in range(len(input_neurons)):
            self.back_weights[i] = input_neurons[i].weights[index]
   
    def calculate_total_backinput(self):
        total = 0.0
        for i in range(len(self.back_inputs)):
            total += self.back_inputs[i] * self.back_weights[i]
        return total

    def calculate_backoutput(self,back_inputs):
        self.back_inputs = back_inputs
        self.back_output = self.calculate_dy_dz()*self.calculate_total_backinput()
        return self.back_output

    def calculate_dloss_dw(self,batch_index):
        #print(self.d_weights)
        for i in range(len(self.inputs)):
            self.d_weights[i][batch_index] = self.back_output * self.calculate_dz_dw(i)

    def calculate_dloss_db(self,batch_index):
        self.d_bias[batch_index] = self.back_output * self.calculate_dz_db()

    def update_weights(self):
        #print (self.d_weights)
        for i in range(len(self.inputs)):
            #print (self.d_weights[i])
            self.weights[i] = self.weights[i] - 0.15 * sum(self.d_weights[i])
            self.bias = self.bias - 0.15 * sum(self.d_bias)


class NeuralLayer:
    
    def __init__(self,num_neurons):
        self.bias = random.random()
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
    
    def init_weights(self,nums_inputs):
        for neuron in self.neurons:
            neuron.init_weights(nums_inputs)

    def set_mini_batch(self,mini_batch):
        for neuron in self.neurons:
            neuron.set_mini_batch(mini_batch)

    def feed_forward(self,inputs):
        outputs = []
        self.input_x = list(inputs)
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(self.input_x))
        return outputs

    def get_outputs(self):
        outputs = [0] * len(self.neurons)
        for i in range(len(self.neurons)):
            outputs[i] = self.neurons[i].output
        return outputs
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs
  
    def flash_back_weights(self,last_layer):
        for i in range(len(self.neurons)):
            self.neurons[i].flash_back_weights(i,last_layer.neurons)

    def calculate_back_inputs(self,training_output):
        back_inputs = []
        back_input  = 0.0

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            back_input = neuron.calculate_dloss_dy(training_output[i]) * neuron.calculate_dy_dz()
            back_inputs.append(back_input)
            neuron.back_output = back_input
		
        return back_inputs

           
    def back_forward(self,back_inputs):
        back_outputs = []
        self.back_input_x = list(back_inputs)
        for neuron in self.neurons:
            back_outputs.append(neuron.calculate_backoutput(self.back_input_x))
        return back_outputs

    def calculate_dloss(self,batch_index):
	for neuron in self.neurons:
            neuron.calculate_dloss_dw(batch_index)
            neuron.calculate_dloss_db(batch_index)

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()    
                     

class NeuralNetwork:
     
    def __init__(self,num_inputs,num_outputs):
        self.num_inputs   = num_inputs
        self.output_layer = NeuralLayer(num_outputs) 
        self.hiden_layers = []     

    def add_hiden_layer(self,num_neurons):
	
        layer = NeuralLayer(num_neurons) 
        if(len(self.hiden_layers) == 0):
           layer.init_weights(self.num_inputs)
        else:
           layer.init_weights(len(self.hiden_layers[-1].neurons))
           self.hiden_layers[-1].flash_back_weights(layer)

        self.output_layer.init_weights(num_neurons)
        layer.flash_back_weights(self.output_layer)
        self.hiden_layers.append(layer)

    def calculate_back_inputs(self,training_output):
		
        return self.output_layer.calculate_back_inputs(training_output)

    def feed_forward(self,inputs):
        #print(inputs)
        input_x = list(inputs)
        for hiden_layer in self.hiden_layers:
            out_puts  = hiden_layer.feed_forward(input_x)
            input_x   = out_puts
        return self.output_layer.feed_forward(out_puts)

    def back_forward(self,training_output):
        back_inputs = self.calculate_back_inputs(training_output)
        back_input_x = list(back_inputs)
        for back_hiden_layer in reversed(self.hiden_layers):
            backout_puts = back_hiden_layer.back_forward(back_input_x)
            back_input_x = backout_puts
    
    def calculate_dloss(self,batch_index):
        for layer in self.hiden_layers:
            layer.calculate_dloss(batch_index)
        self.output_layer.calculate_dloss(batch_index)

    def update_parameters(self):
        for layer in self.hiden_layers:
            layer.update_weights()
        self.output_layer.update_weights()
        self.flash_back_parameter()

    def flash_back_parameter(self):
        for i in range(len(self.hiden_layers)):
            layer = self.hiden_layers[i]
            if (i+1) < len(self.hiden_layers):
               layer.flash_back_weights(self.hiden_layers[i+1])
            else:
               layer.flash_back_weights(self.output_layer)

    def set_mini_batch(self,mini_batch):
        for layer in self.hiden_layers:
            layer.set_mini_batch(mini_batch)
        self.output_layer.set_mini_batch(mini_batch)

    def train(self,training_inputs,training_output,epoches,mini_batch):
        self.set_mini_batch(mini_batch)
        for epoch in range(epoches):
            for i in range(mini_batch):
                index = (epoch + i) % len(training_inputs)
                self.feed_forward(training_inputs[index])
                self.back_forward(training_output[index])
                self.calculate_dloss(i)
            self.update_parameters()

    def predit(self,predit_inputs):
        self.feed_forward(predit_inputs)
        return self.output_layer.get_outputs()

def show_weights(nn):
    for layer in nn.hiden_layers:
        for neuron in layer.neurons:
            print ('weights:')
            print (neuron.weights)
            print ('back_weights:')
            print (neuron.back_weights)
    for neuron in nn.output_layer.neurons:
        print ('out_put weights:')
        print (neuron.weights)
        print ('output back_weights:')
        print (neuron.back_weights)

x_label = [[0,0],[0,1],[1,0],[1,1]]
y_label = [[0],[1],[1],[0]]

nn = NeuralNetwork(2,1)
nn.add_hiden_layer(5)
nn.train(x_label, y_label,2000,1)

for t in range(4):
    print(nn.predit(x_label[t]))
 

