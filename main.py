import os
import numpy as np
import random
# these libraries are used for statistical analysis purposes
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

def load_digits_data_label():

    # loading files according to their names and storing them in numpy arrays
    digit_test_0 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_0.txt"), delimiter = ',')
    digit_test_1 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_1.txt"), delimiter = ',')
    digit_test_2 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_2.txt"), delimiter = ',')
    digit_test_3 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_3.txt"), delimiter = ',')
    digit_test_4 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_4.txt"), delimiter = ',')
    digit_test_5 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_5.txt"), delimiter = ',')
    digit_test_6 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_6.txt"), delimiter = ',')
    digit_test_7 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_7.txt"), delimiter = ',')
    digit_test_8 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_8.txt"), delimiter = ',')
    digit_test_9 = np.genfromtxt(os.path.join('a1digits-data',"digit_test_9.txt"), delimiter = ',')
    digit_train_0 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_0.txt"), delimiter = ',')
    digit_train_1 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_1.txt"), delimiter = ',')
    digit_train_2 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_2.txt"), delimiter = ',')
    digit_train_3 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_3.txt"), delimiter = ',')
    digit_train_4 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_4.txt"), delimiter = ',')
    digit_train_5 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_5.txt"), delimiter = ',')
    digit_train_6 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_6.txt"), delimiter = ',')
    digit_train_7 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_7.txt"), delimiter = ',')
    digit_train_8 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_8.txt"), delimiter = ',')
    digit_train_9 = np.genfromtxt(os.path.join('a1digits-data',"digit_train_9.txt"), delimiter = ',')

    # creating labels in one hot encoding format where the index of 1 represent that class.
    label_digit_test_0 = np.tile([1,0,0,0,0,0,0,0,0,0], (digit_test_0.shape[0],1))
    label_digit_test_1 = np.tile([0,1,0,0,0,0,0,0,0,0], (digit_test_1.shape[0],1))
    label_digit_test_2 = np.tile([0,0,1,0,0,0,0,0,0,0], (digit_test_2.shape[0],1))
    label_digit_test_3 = np.tile([0,0,0,1,0,0,0,0,0,0], (digit_test_3.shape[0],1))
    label_digit_test_4 = np.tile([0,0,0,0,1,0,0,0,0,0], (digit_test_4.shape[0],1))
    label_digit_test_5 = np.tile([0,0,0,0,0,1,0,0,0,0], (digit_test_5.shape[0],1))
    label_digit_test_6 = np.tile([0,0,0,0,0,0,1,0,0,0], (digit_test_6.shape[0],1))
    label_digit_test_7 = np.tile([0,0,0,0,0,0,0,1,0,0], (digit_test_7.shape[0],1))
    label_digit_test_8 = np.tile([0,0,0,0,0,0,0,0,1,0], (digit_test_8.shape[0],1))
    label_digit_test_9 = np.tile([0,0,0,0,0,0,0,0,0,1], (digit_test_9.shape[0],1))
    label_digit_train_0 = np.tile([1,0,0,0,0,0,0,0,0,0], (digit_train_0.shape[0],1))
    label_digit_train_1 = np.tile([0,1,0,0,0,0,0,0,0,0], (digit_train_1.shape[0],1))
    label_digit_train_2 = np.tile([0,0,1,0,0,0,0,0,0,0], (digit_train_2.shape[0],1))
    label_digit_train_3 = np.tile([0,0,0,1,0,0,0,0,0,0], (digit_train_3.shape[0],1))
    label_digit_train_4 = np.tile([0,0,0,0,1,0,0,0,0,0], (digit_train_4.shape[0],1))
    label_digit_train_5 = np.tile([0,0,0,0,0,1,0,0,0,0], (digit_train_5.shape[0],1))
    label_digit_train_6 = np.tile([0,0,0,0,0,0,1,0,0,0], (digit_train_6.shape[0],1))
    label_digit_train_7 = np.tile([0,0,0,0,0,0,0,1,0,0], (digit_train_7.shape[0],1))
    label_digit_train_8 = np.tile([0,0,0,0,0,0,0,0,1,0], (digit_train_8.shape[0],1))
    label_digit_train_9 = np.tile([0,0,0,0,0,0,0,0,0,1], (digit_train_9.shape[0],1))

    # concatenate all test and label data together in the order of labels, they will also be shuffled in the same order later.
    digit_dataset_test = np.concatenate((digit_test_0, digit_test_1, digit_test_2, digit_test_3, digit_test_4, digit_test_5, digit_test_6, digit_test_7, digit_test_8, digit_test_9), axis = 0)
    digit_label_test = np.concatenate((label_digit_test_0, label_digit_test_1, label_digit_test_2, label_digit_test_3, label_digit_test_4, label_digit_test_5, label_digit_test_6, label_digit_test_7, label_digit_test_8, label_digit_test_9), axis = 0)
    digit_dataset_train = np.concatenate((digit_train_0, digit_train_1, digit_train_2, digit_train_3, digit_train_4, digit_train_5, digit_train_6, digit_train_7, digit_train_8, digit_train_9), axis = 0)
    digit_label_train = np.concatenate((label_digit_train_0, label_digit_train_1, label_digit_train_2, label_digit_train_3, label_digit_train_4, label_digit_train_5, label_digit_train_6, label_digit_train_7, label_digit_train_8, label_digit_train_9), axis = 0)

    # shuffle dataset and labels in sync
    test_indices = np.arange(digit_dataset_test.shape[0])
    np.random.shuffle(test_indices)
    digit_dataset_test = digit_dataset_test[test_indices]
    digit_label_test = digit_label_test[test_indices]

    train_indices = np.arange(digit_dataset_train.shape[0])
    np.random.shuffle(train_indices)
    digit_dataset_train = digit_dataset_train[train_indices]
    digit_label_train = digit_label_train[train_indices]

    return digit_dataset_test, digit_label_test, digit_dataset_train, digit_label_train

def load_cancer_data_label():
    cancer_data = np.delete(np.genfromtxt("breast-cancer-wisconsin.data", delimiter = ','),0,1)
    # remove nans or ? data which is all in column 5 and replace with mean of the column
    a = np.argwhere(np.isnan(cancer_data))
    b = int(np.nanmean(cancer_data[:,5]))
    for i in range(len(a)):
        cancer_data[a[i][0],a[i][1]] = b
    # same shape of labels as digits
    cancer_label = cancer_data[:,-1].reshape((cancer_data[:,-1].shape[0],1))
    cancer_label = np.where(cancer_label==2, 0 , 1) # 0 for 2(benign) and 1 for 4(malignant)
    cancer_data = np.delete(cancer_data,-1,1) #delete label row
    # approx 70%-30% train test split, divided by 10 to normalize data
    cancer_dataset_test, cancer_dataset_train = cancer_data[:200]/10, cancer_data[200:]/10
    cancer_label_test, cancer_label_train = cancer_label[:200], cancer_label[200:]
    # shuffle
    test_indices = np.arange(cancer_dataset_test.shape[0])
    np.random.shuffle(test_indices)
    cancer_dataset_test = cancer_dataset_test[test_indices]
    cancer_label_test = cancer_label_test[test_indices]

    train_indices = np.arange(cancer_dataset_train.shape[0])
    np.random.shuffle(train_indices)
    cancer_dataset_train = cancer_dataset_train[train_indices]
    cancer_label_train = cancer_label_train[train_indices]

    return cancer_dataset_test, cancer_label_test, cancer_dataset_train, cancer_label_train

def backprop_network(dataset, label, validation_set, test_set, n_hidden_neurons, n_epochs, activation_function, lr, momentum):

    # number of input and output neurons according to dataset
    n_inputs = len(dataset[0])
    if len(label[0])>2:
        n_outputs = len(label[0])
    else:
        n_outputs = 1

    network = []  # a list to store data of neural network
    #  hidden layer which is a list of neurons with each neuron having weights string as dictionary key and values as weights
    hidden1 = [{'weights':[random.random() for i in range(n_inputs+1)]} for i in range(n_hidden_neurons)] # n_inputs+1 for bias weight
    network.append(hidden1)
    # it will have output neurons
    output_layer = [{'weights':[random.random() for i in range(n_hidden_neurons+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    error_values = []
    # training step
    for epoch in range(n_epochs):
        # vanilla batch backprop, whole network weights are initialized or updated for entire dataset
        error = 0
        for example_index in range(len(dataset)):
            output = forward_step(network, dataset[example_index], activation_function) # gets a list of outputs from each neuron in output layer
            error += sum([(label[example_index][each_class]-output[each_class])**2 for each_class in range(len(label[example_index]))])
            back_prop_step(network, label[example_index], activation_function, momentum)
            # weights update step
            for layer_index in range(len(network)):
                row = dataset[example_index]
                if layer_index != 0: # if layer is not input layer, input is the output of previous connecting neuron
                    row = [neuron['output'] for neuron in network[layer_index-1]]
                for neuron in network[layer_index]:
                    neuron['weights'][-1] += lr*neuron['errGrad'] # for each neuron, update the bias weight
                    for input_neuron in range(len(row)): # each input neuron from example
                        neuron['weights'][input_neuron] += lr*neuron['errGrad']*row[input_neuron]
        error_values.append(error)
        # shuffle after every epoch
        #set_indices = np.arange(dataset.shape[0])
        #np.random.shuffle(set_indices)
        #dataset = dataset[set_indices]
        #label = label[set_indices]
    predictions = []
    predictions_test_set = []
    for example in validation_set:
        # gets a prediction value in one-hot encoding format
        prediction = predict(network, example, activation_function)
        predictions.append(prediction)
    for example in test_set:
        # gets a prediction value in one-hot encoding format
        prediction_test = predict(network, example, activation_function)
        predictions_test_set.append(prediction_test)
    return predictions, predictions_test_set, error_values


def forward_step(network, example, activation_function):
    input_values = example
    for layer in network: # starts with 1st hidden and then output
        input_to_outputlayer = []
        #if layer_index == 1: # if output layer then sigmoid activation
         #   activation_function = sigmoid
        for neuron in layer: # for each neuron in the current layer
            weighted_sum_value = weighted_sum(input_values, neuron['weights'])
            # creates a new key 'output' to store the the output value of neuron
            neuron['output'] = activation_function(weighted_sum_value)
            input_to_outputlayer.append(neuron['output'])
        input_values = input_to_outputlayer # output of one example
    output = input_values
    return output # list of all outputs from output layer


def weighted_sum(input_values, weights):
    weighted_sum_value = weights[-1]  # last weight of neuron which is bias weight
    for x in range(len(weights)-1):
        weighted_sum_value += input_values[x] * weights[x]
    return float(weighted_sum_value)


def sigmoid(weighted_sum_value):
    return 1.0/(1.0+np.exp(-weighted_sum_value))


def tanh(weighted_sum_value):
    return ((np.exp(weighted_sum_value)-np.exp(-weighted_sum_value))/(np.exp(weighted_sum_value)+np.exp(-weighted_sum_value)))


def derivative(activation_function, output):
    if activation_function == sigmoid:
        return output * (1.0-output)
    else:
        return 1-output**2


def back_prop_step(network, label, activation_function, momentum):

    for reverse in reversed(range(len(network))):
        err = []
        prev = 0.0  # previous change
        layer = network[reverse]  # it starts from output layer to input
        if reverse == len(network)-1:  # if output layer
            for neuron_index in range(len(layer)):
                neuron = layer[neuron_index]
                # error at output layer units
                err.append(label[neuron_index]-neuron['output']) # True-output; if class is true is 1 otherwise 0 and output will be a probability value

        else: # other layers
            for neuron_index in range(len(layer)): # index of neuron from current layer
                error_from_outputlayer = 0.0
                for neuron in network[reverse+1]: # neurons from output layer
                    error_from_outputlayer += (neuron['weights'][neuron_index]*neuron['errGrad']) # weight signal from output to current neuron * error calculated at that output neuron
                err.append(error_from_outputlayer)

        for neuron_index in range(len(layer)):
            neuron = layer[neuron_index]
            # storing error gradient in the dictionary of each output neuron and later in hidden
            # add a previous gradient error by multiple of momentum given by user
            neuron['errGrad'] = err[neuron_index]*derivative(activation_function, neuron['output']) + momentum * prev
            prev = neuron['errGrad']


def predict(network, example, activation_function):
    prediction = forward_step(network, example, activation_function)
    return prediction.index(max(prediction))


def cross_valid_split(dataset, label, n_folds):
    dataset_split = []
    label_split = []
    dataset_copy = dataset
    label_copy = label
    fold_size = int(len(dataset_copy)/n_folds) # number of expamples in one fold
    for fold_index in range(n_folds):
        # create folds in sync with dataset and labels on same index
        dataset_fold = np.empty((1,len(dataset_copy[0]))) # np array to store random example
        label_fold = np.empty((1,len(label_copy[0])))
        while len(dataset_fold)<fold_size:
            # select randomly from already shuffled data
            #if len(dataset_copy) != 0:
            example_index = random.randrange(len(dataset_copy))
            dataset_fold = np.vstack((dataset_fold, dataset_copy[example_index]))
            dataset_copy = np.delete(dataset_copy, example_index,0) # delete the example from copy so it is not used in different fold
            label_fold = np.vstack((label_fold, label_copy[example_index]))
            label_copy = np.delete(label_copy, example_index,0)
        dataset_fold = np.delete(dataset_fold, 0,0) # delete the empty row which was created to initialize array
        label_fold = np.delete(label_fold, 0,0) # delete the empty row which was created to initialize array
        # append the numpy array in the folds list
        dataset_split.append(dataset_fold)
        label_split.append(label_fold)
    return dataset_split, label_split


def test(n_hidden_neurons, n_epochs, activation_function, lr, momentum,c):
    if activation_function == 1:
        activation_function = sigmoid
    if activation_function == 2:
        activation_function = tanh
    digit_dataset_test, digit_label_test, digit_dataset_train, digit_label_train = load_digits_data_label()
    # k-fold cross vaildation for digits
    n_folds = 4
    digit_dataset_train_folds, digit_label_train_folds = cross_valid_split(digit_dataset_train, digit_label_train, n_folds)
    results_validation = []
    results_test = []
    acc_error = []
    for fold_index in range(len(digit_dataset_train_folds)): # range of this loop is number of folds
        digit_ktrain_set = np.empty((1,len(digit_dataset_train[0]))) # np array to merge all k-1 folds
        digit_klabel_set = np.empty((1,len(digit_label_train[0])))
        # use fold index as index of fold to be used as validation set
        for set_index in range(len(digit_dataset_train_folds)):
            # all folds except current fold index,so loop will be (1,2,3,4), (0,2,3,4),(0,1,3,4).. and left out fold will be validation set
            if set_index != fold_index:
                digit_ktrain_set = np.vstack((digit_ktrain_set, digit_dataset_train_folds[set_index])) # vertically stack all k-1 folds
                digit_klabel_set = np.vstack((digit_klabel_set, digit_label_train_folds[set_index]))
        digit_ktrain_set = np.delete(digit_ktrain_set, 0,0) # delete 1st row which is created during initializaton
        digit_klabel_set = np.delete(digit_klabel_set, 0,0)
        digit_validation_set = digit_dataset_train_folds[fold_index] # validation set
        digit_validation_label = digit_label_train_folds[fold_index] # for accuracy measurements
        print('Running---------')
        print("Fold no.: ", fold_index)
        predictions, predictions_test_set, errors = backprop_network(dataset = digit_ktrain_set, label = digit_klabel_set, validation_set = digit_validation_set, test_set=digit_dataset_test, n_hidden_neurons= n_hidden_neurons, n_epochs = n_epochs, activation_function = activation_function, lr = lr, momentum = momentum)
        actual = [np.argmax(true_value) for true_value in digit_validation_label] # index of maximum value(1) in validation labels
        print("Accumulated sum error over epochs: ", errors)
        print()
        acc_error.append(errors)
        correct = 0
        for i in range(len(predictions)):
            if actual[i] == predictions[i]:
                correct += 1
        results_validation.append(correct/float(len(actual))*100)
        print("Correct classification validation set: ", correct)
        print()
        correct_test = 0
        actual_test = [np.argmax(true_value) for true_value in digit_label_test]
        for i in range(len(predictions_test_set)):
            if actual_test[i] == predictions_test_set[i]:
                correct_test += 1
        results_test.append(correct_test/float(len(actual_test))*100)
        print("Correct classification test set: ", correct_test)
        print("end of fold")
        print()
    mean_result_valid = sum(results_validation)/float(len(results_validation))
    print("Fold test accuracies%: ", results_test)
    print("Fold validation accuracies%: ", results_validation)
    print("Mean fold validation accuracies%: ", mean_result_valid)

    # plotting error over epoch for 4 folds if more folds add another axis for plotting
    title = 'Accumulated sum error over epoch in fold (using hidden layer:', n_hidden_neurons, 'epochs:', n_epochs,'Learning Rate', lr, 'Momentum:', momentum
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(title, fontsize=7)
    ax1.plot(acc_error[0])
    ax1.set_title('Fold 1')
    ax2.plot(acc_error[1])
    ax2.set_title('Fold 2')
    ax3.plot(acc_error[2])
    ax3.set_title('Fold 3')
    ax4.plot(acc_error[3])
    ax4.set_title('Fold 4')
    for ax in fig.get_axes():
        ax.label_outer()
        ax.set(xlabel='Number of epochs', ylabel='Error')
        ax.grid(True)
    plt.savefig('Errorplot_'+str(c)+ '_' +str(n_hidden_neurons) + '_' + str(n_epochs) + '_' + str(lr) + '_'+ str(momentum)+ '.png')

    return results_test, results_validation


def test2(n_hidden_neurons, n_epochs, activation_function, lr, momentum, c):
    if activation_function == 1:
        activation_function = sigmoid
    if activation_function == 2:
        activation_function = tanh
    cancer_dataset_test, cancer_label_test, cancer_dataset_train, cancer_label_train = load_cancer_data_label()
    # k-fold cross vaildation for cancer
    n_folds = 4
    cancer_dataset_train_folds, cancer_label_train_folds = cross_valid_split(cancer_dataset_train, cancer_label_train, n_folds)
    results_validation = []
    results_test = []
    acc_error = []
    for fold_index in range(len(cancer_dataset_train_folds)): # range of this loop is number of folds
        cancer_ktrain_set = np.empty((1,len(cancer_dataset_train[0]))) # np array to merge all k-1 folds
        cancer_klabel_set = np.empty((1,len(cancer_label_train[0])))
        # use fold index as index of fold to be used as validation set
        for set_index in range(len(cancer_dataset_train_folds)):
            # all folds except current fold index,so loop will be (1,2,3,4), (0,2,3,4),(0,1,3,4).. and left out fold will be validation set
            if set_index != fold_index:
                cancer_ktrain_set = np.vstack((cancer_ktrain_set, cancer_dataset_train_folds[set_index])) # vertically stack all k-1 folds
                cancer_klabel_set = np.vstack((cancer_klabel_set, cancer_label_train_folds[set_index]))
        cancer_ktrain_set = np.delete(cancer_ktrain_set, 0,0) # delete 1st row which is created during initializaton
        cancer_klabel_set = np.delete(cancer_klabel_set, 0,0)
        cancer_validation_set = cancer_dataset_train_folds[fold_index] # validation set
        cancer_validation_label = cancer_label_train_folds[fold_index] # for accuracy measurements
        print('Running---------')
        print("Fold no.: ", fold_index)
        predictions, predictions_test_set, errors = backprop_network(dataset = cancer_ktrain_set, label = cancer_klabel_set, validation_set = cancer_validation_set, test_set = cancer_dataset_test, n_hidden_neurons= n_hidden_neurons, n_epochs = n_epochs, activation_function = activation_function, lr = lr, momentum = momentum)
        actual = [np.argmax(true_value) for true_value in cancer_validation_label] # index of maximum value(1) in validation labels
        print("Accumulated sum error over epochs: ", errors)
        print()
        acc_error.append(errors)
        correct = 0
        for i in range(len(predictions)):
            if actual[i] == predictions[i]:
                correct += 1
        results_validation.append(correct/float(len(actual))*100)
        print("Correct classification validation set: ",correct)
        print()
        correct_test = 0
        actual_test = [np.argmax(true_value) for true_value in cancer_label_test]
        for i in range(len(predictions_test_set)):
            if actual_test[i] == predictions_test_set[i]:
                correct_test += 1
        results_test.append(correct_test/float(len(actual_test))*100)
        print("Correct classification test set: ", correct_test)
        print("end of fold")
        print()
    mean_result_valid = sum(results_validation)/float(len(results_validation))
    print("Fold test accuracies%: ", results_test)
    print("Fold validation accuracies%: ", results_validation)
    print("Mean fold validation accuracies%: ", mean_result_valid)

    # plotting error over epoch for 4 folds if more folds add another axis for plotting
    title = 'Accumulated sum error over epoch in fold (using hidden layer:', n_hidden_neurons, 'epochs:', n_epochs,'Learning Rate', lr, 'Momentum:', momentum

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(title, fontsize=7)
    ax1.plot(acc_error[0])
    ax1.set_title('Fold 1')
    ax2.plot(acc_error[1])
    ax2.set_title('Fold 2')
    ax3.plot(acc_error[2])
    ax3.set_title('Fold 3')
    ax4.plot(acc_error[3])
    ax4.set_title('Fold 4')
    for ax in fig.get_axes():
        ax.label_outer()
        ax.set(xlabel='Number of epochs', ylabel='Error')
        ax.grid(True)
    plt.savefig('Errorplot_' +str(c)+ '_' +str(n_hidden_neurons) + '_' + str(n_epochs) + '_' + str(lr) + '_'+ str(momentum)+ '.png')

    return results_test, results_validation


# statistical analysis using python libraries
def statanalysis(test_accuracy, valid_accuracy, c):
    #len(test_accuracy) # no of tests
    #len(test_accuracy[0]) # no of folds
    test_dic = {'test'+str(num_test): test_value for num_test, test_value in enumerate(test_accuracy)}
    valid_dic = {'test'+str(num_test): test_value for num_test, test_value in enumerate(valid_accuracy)}
    df_test = pd.DataFrame(test_dic)
    df_valid = pd.DataFrame(valid_dic)
    print()
    print("Test results: ")
    print(df_test)
    print("Test stats: ")
    print(df_test.describe())
    print()
    print("Validation results: ")
    print(df_valid)
    print("Valid stats: ")
    print(df_valid.describe())

    fig1, ax1 = plt.subplots() # boxplot of test accuracies for initial reference
    ax1.set_title('Test accuracy Plot')
    ax1.boxplot(df_test)
    plt.savefig('BoxTest'+str(c)+ '_'+str(test_dic['test1'][0])+'.png') # store with name first value of test

    # shapiro test to test null hypothesis if accuracies came from normally distributed population
    # if the resulting p-value greater than 0.05 then it is normally distribute
    # otherwise the hypothesis is rejected
    print()
    print("shapiro test for testing accuracies")
    for col, val in df_test.iteritems():
        print(col, end=' ')
        stat, p = shapiro(val)
        print('stats=%.4f, p=%.4f' % (stat,p))
    print()
    print("shapiro test for validation accuracies")
    for col, val in df_valid.iteritems():
        print(col, end=' ')
        stat, p = shapiro(val)
        print('stats=%.4f, p=%.4f' % (stat, p))

    # ANOVA tests which tests the hypothesis that mean of datasets are not statistically different from each other
    # Hypothesis is rejected if p-value is less than 0.05 and there exist a better mean value among test sets
    print()
    print("ANOVA test")
    a_model = ols('df_test.iloc[:,0]~df_test.iloc[:,1]+df_test.iloc[:,2]', df_test).fit()
    anova_table = sm.stats.anova_lm(a_model, typ=2)
    print(anova_table)

    # t-test which tests our hypothesis that all means are same and there exist no statistical advantage over any parameter
    # if p-value is less than 0.05 then there is statistical evidence that one of the mean is better and earlier hypothesis is rejected

    print()
    print("t-tests: ")
    t1 = ttest_ind(df_test.iloc[:,0],df_test.iloc[:,1], equal_var=False)
    t2 = ttest_ind(df_test.iloc[:,1],df_test.iloc[:,2], equal_var=False)
    t3 = ttest_ind(df_test.iloc[:,0],df_test.iloc[:,2], equal_var=False)

    print(t1)
    print(t2)
    print(t3)
    print()

    # comparison of p-values
    if t1[1]< 0.005 or t2[1]< 0.005 or t3[1]< 0.005:
        print("Statistically better mean available, the one with highest average accuracy")
    else:
        print("No statistically better mean")


def main():
    a = int(input("Enter 1 for Digits classification, Enter 2 for Cancer classification: "))
    b = int(input("Enter number of tests to compare (only works on 3), if don't want to run tests just single run enter 1: "))
    c = input("Give name to your Test for files: ")
    if b == 3:
        test_accu = []
        valid_accu = []
        for i in range(b):
            print()
            print("Test no.: "+str(i))
            if a == 1:
                n_hidden_neurons= int(input("Enter number of neurons for hidden layer: "))
                n_epochs = int(input("Enter number of epochs: "))
                activation_function = int(input("Enter 1 for sigmoid or 2 for tanh activation function: "))
                lr = float(input("Enter learning rate: "))
                momentum = float(input("Enter Momentum: "))
                results_test, results_validation = test(n_hidden_neurons, n_epochs, activation_function, lr, momentum, c)
                test_accu.append(results_test)
                valid_accu.append(results_validation)

            if a == 2:
                n_hidden_neurons= int(input("Enter number of neurons for hidden layer: "))
                n_epochs = int(input("Enter number of epochs: "))
                activation_function = int(input("Enter 1 for sigmoid or 2 for tanh activation function: "))
                lr = float(input("Enter learning rate: "))
                momentum = float(input("Enter Momentum: "))
                results_test, results_validation = test2(n_hidden_neurons, n_epochs, activation_function, lr, momentum, c)
                test_accu.append(results_test)
                valid_accu.append(results_validation)
        statanalysis(test_accu, valid_accu, c)

    # normal run without statistical analysis
    if b == 1:
        if a == 1:
            n_hidden_neurons= int(input("Enter number of neurons for hidden layer: "))
            n_epochs = int(input("Enter number of epochs: "))
            activation_function = int(input("Enter 1 for sigmoid or 2 for tanh activation function: "))
            lr = float(input("Enter learning rate: "))
            momentum = float(input("Enter Momentum: "))
            results_test, results_validation = test(n_hidden_neurons, n_epochs, activation_function, lr, momentum, c)

        if a == 2:
            n_hidden_neurons= int(input("Enter number of neurons for hidden layer: "))
            n_epochs = int(input("Enter number of epochs: "))
            activation_function = int(input("Enter 1 for sigmoid or 2 for tanh activation function: "))
            lr = float(input("Enter learning rate: "))
            momentum = float(input("Enter Momentum: "))
            results_test, results_validation = test2(n_hidden_neurons, n_epochs, activation_function, lr, momentum, c)



if __name__ == '__main__' :
    main()