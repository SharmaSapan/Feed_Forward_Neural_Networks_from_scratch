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


