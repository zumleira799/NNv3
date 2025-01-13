# NNv3
 I made a simple ML model from scratch and trained it on the mnist dataset

This is a simple cuda script that alows you to generate any 4 layer feed-forwarding neural network and train it using ADAM,
you can use your own custom training data if you want, as long as you update the first value in "LayerVals", but the true result always needs to be in position 0 of each training image
In the future I want to make the learning more general and allow the program to do training with more than 4 layers

For this to work you need to run it on linux, bcs of the assembly script, you need nvcc installed, and you need a GPU that supports dynamic parellellism
If you want to change some meta parameters the ones that currently exist and all the ones that will exist in the future will be/are defined as macros

About the model performance, I managed to make it go from a 10% accuracy to around 90/95% in abt 3 epochs, the loss function I use is cross-entropy and it goes from around 2.3 to around 0.2 also in abt 3 epochs (with a learning rate of 0.001), this is with the default neural net configuration in the file, 2 hidden layers of 16 neurons each, and an output layer of 10

I use Leaky ReLU as an activation function in the hidden layers and softmax as the final activation function