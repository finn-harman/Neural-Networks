import numpy as np
import pickle
import torch
import torch.nn as nn  # importing pytorch
import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import *
import matplotlib.pyplot as plt
import matplotlib as mtp


class ClaimClassifier(nn.Module):  # inheriting from nn.Module


    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimClassifier,
              self).__init__()  # combined with inheriting from nn.Module, this creates a class that tracks the
        self.layers = None
        self.maximum = None
        self.minimum = None
        self.auc = None
        self.tpr = None
        self.fpr = None
        self.threshold = None
        pass

    def forward(self, x):
        # x = torch.tanh(self.hidden1(x))  # seems to be a common combo of activations
        # # x = torch.tanh(self.hidden2(x))
        # x = torch.sigmoid(self.output(x))

        return self.layers(x)

    def _preprocessor(self, X_raw, sampling_type=None):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """

        # if sampling type given, perform down or up sampling on dataset
        if (sampling_type != None):
            count_class_0, count_class_1 = X_raw.made_claim.value_counts()
            train_class_0 = X_raw[X_raw['made_claim'] == 0]
            train_class_1 = X_raw[X_raw['made_claim'] == 1]
            if (sampling_type == "over"):
                print("Performing over-sampling on minority class")
                train_class_1_over = train_class_1.sample(count_class_0, replace=True)
                X_raw = pd.concat([train_class_0, train_class_1_over], axis=0)
                print(X_raw.made_claim.value_counts())
            if (sampling_type == "under"):
                print("Performing under-sampling on majority class")
                train_class_0_under = train_class_0.sample(count_class_1)
                X_raw = pd.concat([train_class_0_under, train_class_1], axis=0)
                print(X_raw.made_claim.value_counts())

        # splitting data into inputs and labels
        X_cleaned = X_raw.drop(columns=["claim_amount", "made_claim"])
        y_cleaned = X_raw["made_claim"]

        # normalising each column in the date
        if self.maximum is None and self.minimum is None:
            # saving min/max values for normalisation of test data
            self.minimum = X_cleaned.min()
            self.maximum = X_cleaned.max()

        X_normalised = (X_cleaned - self.minimum) / (self.maximum - self.minimum)

        # converting from dataframe to numpy
        X_normalised_numpy = X_normalised.to_numpy()
        y_numpy = y_cleaned.to_numpy()

        # converting array to float values
        X_final = X_normalised_numpy.astype(float)
        y_final = y_numpy.astype(float)

        return X_final, y_final  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, layers=None, in_batches=140, epochs=1000,
            learning_rate=0.016, sampling="over"):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """


        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        # Split data to training and validation data
        train, validation = train_test_split(X_raw, test_size=0.2)

        # cleaning data (normalise , convert to numpy, data to float), choosing over-sampling
        X_train, y_train = self._preprocessor(train, sampling)
        X_validation, y_validation = self._preprocessor(validation)

        #Set up layers
        input_attributes = len(X_train[0])

        if layers == None:
            layers = [nn.Linear(input_attributes, 1), nn.Tanh(), nn.Linear(1, 1), nn.Sigmoid()]
        else:
            layers[0] = nn.Linear(input_attributes, layers[2].in_features)

        self.layers = nn.Sequential(*layers)

        # HYPERPARAMETERS
        loss_function = nn.BCELoss()  # guessing binary cross entropy
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # combining inputs and labels together for batching
        full_dataset = Dataset(X_train, y_train)
        batcher = torch.utils.data.DataLoader(full_dataset, batch_size=in_batches, shuffle=True)

        for epoch in range(epochs):
            if epoch > 0 and epoch % 100 == 0:
                #PRINT batchloss/accuracy information
                print("epoch = % 6d" % epoch, end="\n")
                print("batchloss = % 7.4f" % loss.item(), end ="\n")
                acc = self.accuracy(X_validation, y_validation)
                print("accuracy = % 0.2f " % acc)
                X_validation_temp = torch.Tensor(X_validation)
                temp_output = self(X_validation_temp)
                temp_predictions = (temp_output > 0.5).float()
                print(type(temp_output))
                print(type(temp_predictions))
                print("auc = % 0.6f " % self.auc)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # looping through batches
            for inputs, labels in batcher:
                optimizer.zero_grad()
                inputs = inputs.float()
                output = self(inputs)

                loss = loss_function(output.squeeze(), labels)
                loss.backward()
                optimizer.step()

            temp_output = self(torch.Tensor(X_validation))
            current_auc = sk.metrics.roc_auc_score(y_validation, temp_output.detach().numpy())
            # current_acc = self.accuracy(X_validation, y_validation)

            # some validation about which model to use...
            if (epoch == 0):
                self.auc = current_auc
                # self.acc = current_acc
                counter = 1
            elif (self.auc < current_auc):
                self.auc = current_auc
                # self.acc = current_acc
                pred_val = self(torch.Tensor(X_validation))
                self.determine_optimal_threshold(pred_val, y_validation)
                self.save_optimal_model()
                counter = 0
            else:
                counter += 1

            if counter > 20:  # if no improvements afet 20 epochs, end training
                break

        # Set Self to optimal model
        self = load_optimal_model()
        # print("We get here!")
        # print("Final auc: ", self.auc)
        print("Chosen Threshold: ", self.threshold)
        # self.plot_final_roc_curve()
        print("Finished at epoch: ", epoch)
        print("Evaluation using validation dataset: ")
        raw_pred = self(torch.Tensor(X_validation))
        raw_pred = raw_pred.detach().numpy()
        self.evaluate_architecture(raw_pred, y_validation)


        return self

    def accuracy(self, data_x, data_y):
        # data_x and data_y are numpy array-of-arrays matrices

        X = torch.Tensor(data_x)
        Y = torch.ByteTensor(data_y)
        output = self(X)
        pred_y = (output>=0.5)
        acc = sk.metrics.accuracy_score(data_y, np.array(pred_y))

        return acc

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        X_normal = (X_raw - self.minimum) / (self.maximum - self.minimum)
        X_normal = X_normal.to_numpy()
        X_normal = X_normal.astype(float)

        test_data = torch.Tensor(X_normal)
        output = self(test_data)
        predictions_numpy = output.detach().numpy()

        return predictions_numpy  # YOUR PREDICTED CLASS LABELS

    def determine_optimal_threshold(self, raw_predictions, labels):
        # Determine threshold
        self.fpr, self.tpr, thresholds = sk.metrics.roc_curve(labels, raw_predictions.detach().numpy())
        max_index = np.argmax(self.tpr - self.fpr)
        self.threshold = thresholds[max_index]

    def evaluate_architecture(self, predictions, labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        if self.threshold == None:
            self.threshold = 0.5
        predictions_based_on_threshold = np.where(predictions >= self.threshold, 1.0, 0)
        accuracy = sk.metrics.accuracy_score(np.array(labels), np.array(predictions_based_on_threshold))
        auc = sk.metrics.roc_auc_score(labels, predictions)
        print("accuracy = % 0.2f" % accuracy)
        print("auc score = %0.2f" % auc)
        report = sk.metrics.classification_report(np.array(labels), np.array(predictions_based_on_threshold),
                                                  target_names=['class 0', 'class 1'])
        print(report)

    def plot_final_roc_curve(self):
        plt.plot(self.fpr, self.tpr)
        plt.show()

    def save_model(self):
        # print("save model")
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)

    def save_optimal_model(self):
        # print("save model")
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('optimal_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    # print("load model")
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


def load_optimal_model():
    with open('optimal_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(train, prefix = ""):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    optimisation_results = {}
    print("hyperparameter search")

    #Optimising sampling
    auc_array_sampling, samples = optimiseSampling(train)
    auc_array_sampling_numpy = np.array(auc_array_sampling)
    optimal_index = np.argmax(auc_array_sampling_numpy)
    optimal_sampling = samples[optimal_index]
    saveCurrentNetworkResults([auc_array_sampling, samples], prefix + "optimisation_sampling_results")
    optimisation_results["optimal_sample"] = optimal_sampling

    #Optimising batch size
    auc_array_batches, batches = optimiseBatchSize(train, sample=optimal_sampling)
    auc_array_batches_numpy = np.array(auc_array_batches)
    optimal_index = np.argmax(auc_array_batches_numpy)
    optimal_batch = batches[optimal_index]
    saveCurrentNetworkResults([auc_array_batches, batches], prefix + "optimisation_batches_results")
    saveAucPlot(auc_array_batches, batches, optimal_index, "AUC[-]", "Batch Size[-]",
                "AUC score Against Batch Size", prefix + "AUC_Scores_per_batch", bar_width=8)

    optimisation_results["optimal_batch"] = optimal_batch

    #Optimising for learning rate
    auc_array_lr, learning_rates = optimiseLearningRate(train, batch_size=optimal_batch,
                                                        sample=optimal_sampling)
    auc_array_lr_numpy = np.array(auc_array_lr)
    optimal_index = np.argmax(auc_array_lr_numpy)
    optimal_learning_rate = learning_rates[optimal_index]
    saveCurrentNetworkResults([auc_array_lr, learning_rates], prefix + "optimisation_lr_results")
    saveAucPlot(auc_array_lr, learning_rates, optimal_index,"AUC[-]", "Learning Rate[-]",
                "AUC Score Against Learning Rates", prefix + "AUC_Scores_per_lr", bar_width=0.8)

    optimisation_results["Optimal_learning_rate"] = optimal_learning_rate

    # Optimising neurons for one layer
    auc_array_l1, tpr_array_l1, fpr_array_l1, neurons_l1 = optimiseNeuronsForOneLayer(train,
                                                                          lr= optimal_learning_rate,
                                                                          batch_size=optimal_batch,
                                                                                      sample=optimal_sampling)
    auc_array_l1_numpy = np.array(auc_array_l1)
    optimal_index = np.argmax(auc_array_l1_numpy)
    optimal_neuron_no_l1 = neurons_l1[optimal_index]
    saveRocCurveForNumberOfNeurons(tpr_array_l1, fpr_array_l1, optimal_index, prefix + "Roc_curves_one_layer")
    saveCurrentNetworkResults([auc_array_l1, tpr_array_l1, fpr_array_l1, neurons_l1], prefix + "optimisation_l1_results")
    saveAucPlot(auc_array_l1, neurons_l1, optimal_index,"AUC[-]", "Neurons[-]",
                "AUC Score Against Number of Neurons", prefix + "AUC_Scores_per_l1", bar_width=4)

    optimisation_results["optimal_neuron_no_l1"] = optimal_neuron_no_l1

    # Optimising neurons for two layers
    auc_array_l2, tpr_array_l2, fpr_array_l2, neurons_l2 = optimiseNeuronsForTwoLayers(train, optimal_neuron_no_l1,
                                                                           lr=optimal_learning_rate,
                                                                           batch_size=optimal_batch,
                                                                                       sample=optimal_sampling)
    auc_array_l2_numpy = np.array(auc_array_l2)
    optimal_index = np.argmax(auc_array_l2_numpy)
    optimal_neuron_no_l2 = neurons_l2[optimal_index]
    saveRocCurveForNumberOfNeurons(tpr_array_l2, fpr_array_l2, optimal_index, prefix+"Roc_curves_two_layers")
    saveCurrentNetworkResults([auc_array_l2, tpr_array_l2, fpr_array_l2, neurons_l2], prefix+"optimisation_l2_results")
    saveAucPlot(auc_array_l2, neurons_l2, optimal_index,"AUC[-]", "Neurons[-]",
                "AUC Score Against Number of Neurons", prefix+"AUC_Scores_per_l2", bar_width=4)

    optimisation_results["optimal_neuron_no_l2"] = optimal_neuron_no_l2

    return  optimisation_results

def optimiseSampling(X_raw):
    sampling_methods=["over", "under"]
    auc_array = []
    claimClassifier = ClaimClassifier()
    print("Optimising sampling method")
    for sampling_method in sampling_methods:
        print("Running network with the following sampling method: ", sampling_method)
        claimClassifier.fit(X_raw, sampling=sampling_method)
        auc_array=claimClassifier.auc

    return auc_array, sampling_methods


def optimiseBatchSize(X_raw, sample = "under"):
    min_batch_size = 10
    max_batch_size = 210
    auc_array = []
    batches = []

    print("Optimizing batch size")
    for batch in range(min_batch_size, max_batch_size, 10):
        claimClassifier = ClaimClassifier()
        print("Running network with batch size: ", batch)
        batches.append(batch)
        claimClassifier.fit(X_raw, in_batches=batch, sampling= sample)
        auc_array.append(claimClassifier.auc)


    return auc_array, batches

def optimiseLearningRate(X_raw, batch_size=10, sample="under"):
    current_learning_rate = 0.001
    max_learning_rate = 10
    auc_array = []
    learning_rates = []


    print("Optimising learning Rate")
    while current_learning_rate < max_learning_rate:
        claimClassifier = ClaimClassifier()
        print("Running Network with Learning Rate: ", current_learning_rate)
        learning_rates.append(current_learning_rate)
        claimClassifier.fit(X_raw, learning_rate= current_learning_rate, in_batches=batch_size, sampling=sample)
        auc_array.append(claimClassifier.auc)
        current_learning_rate *= 2

    return auc_array, learning_rates


def optimiseNeuronsForOneLayer(X_raw, lr = 0.01, batch_size = 10, sample="under"):
    max_neurons = 100
    auc_array = []
    tpr_array = []
    fpr_array = []
    neuron_array = []


    print("Optimising no. of neurons (1st layer)")
    for neurons in range(1, max_neurons + 1, 5):
        claimClassifier = ClaimClassifier()
        print("Running Network with no. of neurons (1st layer): ", neurons)
        claimClassifier.fit(X_raw, layers=[nn.Linear(12, neurons),nn.Tanh(), nn.Linear(neurons, 1), nn.Sigmoid()],
                            learning_rate = lr, in_batches=batch_size, sampling=sample)
        auc_array.append(claimClassifier.auc)
        tpr_array.append(claimClassifier.tpr)
        fpr_array.append(claimClassifier.fpr)
        neuron_array.append(neurons)

    return auc_array, tpr_array, fpr_array, neuron_array


def optimiseNeuronsForTwoLayers(X_raw, neurons_for_first_layer, lr = 0.01, batch_size = 10,
                                sample="under" ):
    max_neurons = 100
    auc_array = []
    tpr_array = []
    fpr_array = []
    neuron_array = []


    print("Optimising no. of neurons (2nd layer)")
    for neurons in range(1, max_neurons + 1, 5):
        claimClassifier = ClaimClassifier()
        print("Running Network with no. of neurons (2nd layer): ", neurons)
        claimClassifier.fit(X_raw, layers=[nn.Linear(12, neurons_for_first_layer),
                                           nn.Tanh(), nn.Linear(neurons_for_first_layer, neurons), nn.Tanh(),
                                           nn.Linear(neurons, 1), nn.Sigmoid()], learning_rate=lr,in_batches=batch_size,
                            sampling=sample)

        auc_array.append(claimClassifier.auc)
        tpr_array.append(claimClassifier.tpr)
        fpr_array.append(claimClassifier.fpr)
        neuron_array.append(neurons)

    return auc_array, tpr_array, fpr_array, neuron_array


def saveAucPlot(y_data, x_data, optimal_index, y_label, x_label, title, name, bar_width = 0.8):

    mtp.rc('font', family='times new roman')

    barlist = plt.bar(x_data, y_data, width = bar_width)
    barlist[optimal_index].set_color('r')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(name + ".png", bbox_inches='tight')
    plt.close()


def saveRocCurveForNumberOfNeurons(tpr_array, fpr_array, optimal_index, name):
    mtp.rc('font', family='times new roman')

    print(optimal_index)
    for index in range(len(tpr_array)):
        if (index != optimal_index):
            try:
                plt.plot(fpr_array[index], tpr_array[index], color='b', alpha=0.4)
            except ValueError:
                print("Can not plot for index: ", index)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr_array[optimal_index], tpr_array[optimal_index], 'r')
    plt.ylabel("TPR[-]")
    plt.xlabel("FPR[-]")
    plt.title("ROC curve based on number of neurons")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(name + ".png", bbox_inches='tight')
    plt.close()


def saveCurrentNetworkResults(result, file_name):

    with open(file_name, 'wb') as target:
        pickle.dump(result, target)


def loadNetworkResults(file_name):
    with open(file_name, 'rb') as target:
        network_results = pickle.load(target)
    return network_results
