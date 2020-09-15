The following is an explanation of how to run our code for part 2.

Part 2:
1. To instantiate a ClaimClassifier, you will need to declare and assign 'ClaimClassifier()'
2. To start training a neural network, use ClaimClassifier.fit(x_raw), where x_raw is the training dataset (in panda's
dataframe format) that includes both the attributes and the ground truth labels.
3. You can specify your own neural network structure as an optional argument in ClaimClassifier.fit(x_raw, layers=...).
Other optional arguments include learning rate, batch size, epochs, and sampling method.
4. Once you have trained a neural network. You can then use ClaimClassifier.predict() to predict whether a given
data sample has an insurance claim or not.
5. You can also optimise the hyperparameters of the ClaimClassifier architecture using the
ClaimClassifierHyperParameterSearch(train, prefix = "") function. Where the 'train' parameter is the raw data as
downloaded (in pandas dataframe format) and the prefix is the prefix of the graphs and pickle files that are saved
throughout the optimisation process. This function returns a dictionary of the optimal sampling method, batch size,
learning rate, optimal number of neurons for one layer and two layers.
