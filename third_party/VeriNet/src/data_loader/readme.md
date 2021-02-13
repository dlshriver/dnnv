## Modified NNet 

#### Introduction

The .nnet file format is used for storing Fully connected Relu 
neural networks in a human readable format. The format was first introduced 
for the ACAS Xu network: https://github.com/sisl/NNet and later used in 
several implementations of verification tools, such as Reluplex, ReluVal
and Neurify. However, Neurify also used a  variation of the nnet
format to support convolutional networks, while removing normalization 
information. We will combine the former approaches and also add support for 
and Sigmoid and Tanh activation functions.

#### File format of nnet

The line by line file format:

**1...** Header text. This can be any number of lines so long as they begin with "//"  
**2.** Four values: Number of layers, number of inputs, number of outputs, and maximum layer size  
**3.** A sequence of values describing the network layer sizes. Begin with the input size, 
then the size of the first layer, second layer, and so on until the output layer size   
**4.** Minimum values of inputs (used to keep inputs within expected range) if only one 
value is given, the value is used for all inputs, else one value for each input is given.    
**5.** Maximum values of inputs (used to keep inputs within expected range) if only one 
value is given, the value is used for all inputs, else one value for each input is given.  
**6.** Mean values of inputs (used for normalization). If only one 
value is given, the value is used for all inputs, else one value for each input is given.
**7.** Range values of inputs (used for normalization). If only one 
value is given, the value is used for all inputs, else one value for each input is given.      
**8.** The layers activation functions indicated by one int for each layer.
0 = Relu, 1 = Sigmoid, 2 = Tanh, -1= No activation.  
**9.** The layer types indicated by one int for each layer. 0 = FC, 1 = Conv2d, 2=Batchnorm2d.  
**10...** For each layer; if the layer is a convolutional layer 5 int parameters:  
1: Number of out channels, 2: Number of in channels, 3: kernel size, 4: Stride 5: padding.  
If  the layer is a batch-norm 2d layer:
One line with the running means, then one line with the running variances.  
**11...** Alternating weight and biases vectors, from the first layer to the last.  

Lines 1-3 are exactly as in the original format. Line 4 used to be a flag, but has not 
been been used since before ACAS Xu and is not included in our new format. Our lines 4-7
are as in the original format, but the lines are shifted by one since we removed the flag.
Line 8 is new, and lines 9-10 is in the same format as used by Neurify for covolutional 
networks. The rest is as in the original format.

The files uses "," as separators.

Lines 4-8 are intended to be used for normalization of inputs. The formula for normalization of a input x_i is:  
norm(x_i) = (x_i - mean_i)/ range_i and then clipped to the min/max values in line 4 and 5. 