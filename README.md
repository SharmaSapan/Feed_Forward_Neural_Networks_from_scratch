# Feed_Forward_Neural_Networks_from_scratch

### Neural networks are important to classification problems such as object detection and pattern detection. Feed-forward neural network used in this example explores two types of learning variants on two different sets of data solving different types of classification problem. I tried to find the best hyper-parameters using statistical analysis to classify digits and cancer dataset. Various tests will be concluded and analysed to statistically find the better parameter. ###

The problem on hand are Digit recognition and and Breast Cancer recognition. A popular digit recognition database from the US postal service is used where each file consists of 8*8
pixel given with the labels. The normalized pixel values are given as input to the neural network to learn and validate, then a predicted output of class from 0-9 is given. The neural
networks to solve this problem is interesting as it is not easily solvable using traditional algorithms and it seems easy to humans but not so for machines. Neural network uses large
number of training examples with labels and learn on it by inferring the pattern and adjusting its weight parameters using
back propagation or RMSprop. The data is stored in numpy arrays and labels are created in one hot encoding format
according to file organization. Then all the testing and training data is combined individual to their respective set along with
their labels, in separate set. Data is also shuffled in sync with the labels.

For the second problem, the Wisconsin Breast Cancer Database (WBCD) dataset is used from the UCI machine learning repository. Neural network helps classify if the cancer
is benign or malignant using the 9 dimensions of attributes provided to help predict the class of cancer. The neural network is great for this example as it predicts with high
accuracy by learning from the attributes provided. The data is stored as numpy arrays with single labels predicting 0 for benign and 1 for malignant. There are nan values in data which
is handled by finding mean of that column and replacing those nan by means. The data is normalized to bring all data to scale of 0-1. Labels and data is separated and shuffling is performed
same as in digits data.

Run the main.py file (also has venv environments for pycharm) to test. Most of the hyperparameters are through user input. 
First enter 1 for digits or enter 2 for cancer. Then enter 1 if you don't want to do any tests and enter 3 if you want to do tests and compare 3 tests. Test runs are in test folder containg pdf files and graphs. 

The tests are performed to find if any parameter value of neural network is statistically better than the other. Three tests were performed to check each parameter on high, medium, low range. Each parameter when tested was kept with same other parameters so that results are only due to parameter in question. Parameters that are tested are: Learning rate, Momentum, number of hidden neurons, number of epochs, and activation function. K-fold cross validation method is used to split the data into training and validation data to tune the hyper-parameters in small limited data set to estimate how the model is expected to perform when put to test on unseen data during training of network. It helps us to use more data and get more metrics on the model performance on different data sets and help fine tune the parameters.

Accuracy metrics were used which provides the percentage of correct predictions compared to actual label. Also, sum mean square error was used to see error loss during training and to plot on graph.

Tests are as follows:

### Digits: ###
(test values in bracket)

| Test name  | Hidden Neurons | Epochs | Activation Function | Learning Rate | Momentum |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Learning rate: | 25  | 25| 1| (0.7,1.7,2.7)| 0.15|
|Hidden neurons:| (15,30,45)|25|1| 1.7| 0.15|
|Momentum:|       25  | 25| 1| 1.7| (0.15,0.5,0.85)|
|Epochs: |         25  | (10,30,50)| 1| 1.7| 0.15|
|activation: ||| 1 or 2 for sigmoid and tanh |

### Cancer: ###
(test values in bracket)
| Test name  | Hidden Neurons | Epochs | Activation Function | Learning Rate | Momentum |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|Learning rate:| 3  | 8| 1| (0.03,0.3,3)| 0.5|
|Momentum:  |     3  | 8| 1| 0.5| (0.15,0.5,0.85)|
|Activation: ||| 1 or 2 (sigmoid and tanh)


Error plot for k=4 folds in cancer detection test with 0.03 learning rate:
<img src="https://github.com/SharmaSapan/Feed_Forward_Neural_Networks_from_scratch/blob/master/Tests/Cancer_class/Errorplot_cancer_lr_3_8_0.03_0.5.png" width="700" height="300"> 

Test accuracy Box plot for Digits recogonition when tested with 10, 30 and 50 epochs:
<img src="https://github.com/SharmaSapan/Feed_Forward_Neural_Networks_from_scratch/blob/master/Tests/Digits/BoxTestdigit_epoch_92.55.png" width="400" height="300"> 

### After getting all the results of three parameter test, the dataframe was used to get a glance of all test accuracy result, validation accuracy result and their statistical summary: ###

Test results in %: 
|    |test0|  test1   |test2
| --- | --- | --- | --- |
|0  |88.025 | 92.55  |93.200
|1 | 86.700  |93.50  |93.900
|2  |89.500 | 92.70  |91.925
|3 | 88.850  |93.20 | 93.850

Test stats: 
| |       test0 |     test1|     test2
| --- | --- | --- | --- |
|count  | 4.000000   |4.000000 |  4.00000
|mean|   88.268750 | 92.987500  |93.21875
|std  |   1.207507   |0.440407   |0.91955
|min   | 86.700000  |92.550000  |91.92500
|25%    |87.693750|  92.662500  |92.88125
|50%    |88.437500  |92.950000 | 93.52500
|75%   | 89.012500  |93.275000  |93.86250
|max  |  89.500000  |93.500000 | 93.90000

Validation results in %: 
|    |test0|  test1   |test2
| --- | --- | --- | --- |
|0  |87.764437|  93.253288  |92.224128
|1  |86.163522 | 92.910234  |94.282447
|2 | 89.136650  |93.081761 | 91.652373
|3|  89.594054  |93.710692|  93.253288


Then, Shapiro wilk test is performed on all three test set accuracies.
This test, tests the normality of the data. It tests
the null hypothesis that a sample came from a normally
distributed population. 

ANOVA test is performed on the test accuracy sets on three
different parameters which tells provides us the evidence about
the differences among means. The null hypothesis for the test
is that the means of tests are equal and no significant difference
available. This hypothesis is rejected if the p-values is less than
0.005 and then there exist a test set which has better parameter. 

t-test is performed on accuracy test set to determine if there
is significant difference between the means among all three
tests.t-test which tests our hypothesis that all means are same
and there exist no statistical advantage over any parameter.If
p-value is less than 0.05 then there is statistical evidence that
one of the mean is better and earlier hypothesis is rejected.

Finally, if t-test
rejects hypothesis, the higher accuracy test parameter with
less standard deviation is the best candidate.
