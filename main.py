import pandas as pd
import numpy as np
import random
import os

def find_unique_classes(vect):
    # Find all unique instances in some vector
    # Input: some vector
    # Return: A list of all the unique instances
    classes = []
    for i in range(len(vect)):
        counter = 0
        for j in range(len(classes)):
            if vect[i] == classes[j]:
                pass
            else:
                counter += 1
        if counter == len(classes):
            classes.append(vect[i])
    return classes


def majority_class(vect):
    # Find the class which appears the most in some vector
    # Input: Some vector
    # Return: The majority class
    classes = find_unique_classes(vect)
    class_dict = {}
    majority = 0
    for i in range(len(classes)):
        class_dict[classes[i]] = 0
    for i in range(len(vect)):
        for key in class_dict:
            if key == vect[i]:
                class_dict[key] += 1
    for key in class_dict:
        if class_dict[key] > majority:
            majority = class_dict[key]
            value = key
    return value


def count(vector, value):
    # Count number of instances of a variable in a vector
    # Input: vector = some vector, value = some value
    # Return: number of instances of value in vector
    instances = 0
    for i in range(len(vector)):
        if vector[i] == value:
            instances += 1
    return instances


def c_algorithm(x, y, parent_entropy, node_dict):
    # The bulk of the algorithm occurs here, we populate the decision tree and evaluate the bases cases each time
    # to check if we should terminate. This is a recursive algorithm
    # Input: x = feature array, y = class vector, parent_entropy = entropy of parent node (This will be the global
    # entropy on the first call), node_dict = the node dictionary for this current branch
    # Return: A node which we have creates, it may be a leaf node or a regular node
    print("Starting C4.5 Algorithm...")
    gain_dict = {"Gain Ratio": 0}
    temp_dict = {}
    classes = find_unique_classes(y)
    # Check to see if any of the base cases are true
    if len(classes) == 1:
        print("Only one class left, creating leaf", classes)
        # Only one class left in dataset
        # Create leaf node with that value
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = Node(value=classes[0])
                return node_dict[key]
    elif len(classes) == 0:
        print("No instances in dataset remaining, setting default class as 0")
        # There are no instanes left in the dataset, simply return a default of 0
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = Node(value=0)
                return node_dict[key]
    elif max_all_cols(x) == 0:
        print("All features are 0, creating a leaf node")
        # If all features are 0, ie. we have created nodes for all these features and set their values to 0
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = Node(value=majority_class(y))
                return node_dict[key]
    # The base cases have been checked, now...
    # Loop over each feature
    for i in range(x.shape[1]):
        if column_unused(i, node_dict):  # We must check if this feature has been used in this branch yet, if so we pass
            # Check if discrete or continuous
            if isinstance(x[1, i], str):
                temp_dict = gain_ratio(x[:, i], y, parent_entropy)
                temp_dict["Type"] = "Discrete"
            else:
                temp_dict = gain_ratio_continuous(x[:, i], y, parent_entropy)
                temp_dict["Type"] = "Continuous"
            temp_dict["Index"] = i
            # If current feature provides more info gain...
            if temp_dict["Gain Ratio"] > gain_dict["Gain Ratio"]:
                for key in temp_dict:
                    gain_dict[key] = temp_dict[key]
    # We have checked all features and selected the best one
    # If we have not found any features with an info gain > 0
    # Create a leaf node with the majority class
    if gain_dict["Gain Ratio"] == 0:
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = Node(value=majority_class(y))
                return node_dict[key]
    print("Creating node...")
    if gain_dict["Type"] == "Continuous":  # We handle nodes differently based on their data type
        # Store the indices we need to pass on to each child node
        values_below = []
        values_above = []
        for i in range(len(y)):
            if x[i, gain_dict["Index"]] <= gain_dict["Threshold"]:
                values_below.append(i)
            else:
                values_above.append(i)
        x_below = x[values_below, :]  # For values below/equal to threshold
        y_below = y[values_below]  # For classes below/equal threshold
        x_above = x[values_above, :]  # For values above threshold
        y_above = y[values_above]  # For classes above threshold
        # Set the values of this feature to zero, so they do not contribute any info gain later
        x_below[:, gain_dict["Index"]] = set_values(x_below[:, gain_dict["Index"]], 0)
        x_above[:, gain_dict["Index"]] = set_values(x_above[:, gain_dict["Index"]], 0)
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = gain_dict["Index"]  # To avoid passing an incorrect dictionary in the next step
                node_dict[key] = Node(feat=gain_dict["Index"],
                                      feat_type=gain_dict["Type"],
                                      threshold=gain_dict["Threshold"],
                                      child_1=c_algorithm(x_below, y_below, gain_dict[0], node_dict),  # Recursive call
                                      child_2=c_algorithm(x_above, y_above, gain_dict[1], node_dict))  # Recursive call
                return node_dict[key]  # Return the node we have just created
    else:
        # Store the indices we need to pass on to each child node
        values_0 = []
        values_1 = []
        values_2 = []
        values = find_unique_classes(x[i, gain_dict["Index"]])
        for i in range(len(y)):
            if x[i, gain_dict["Index"]] == values[0]:
                values_0.append(i)
            elif x[i, gain_dict["Index"]] == values[1]:
                values_1.append(i)
            else:
                values_2.append(i)
        x_0= x[values_0, :]
        y_0 = y[values_0]
        x_1 = x[values_1, :]
        y_1 = y[values_1]
        x_2 = x[values_2, :]
        y_2 = y[values_2]
        for key in node_dict:
            if node_dict[key] == "empty":  # Search the node dictionary for an empty key
                node_dict[key] = gain_dict["Index"]  # To avoid passing an incorrect dictionary in the next step
                node_dict[key] = Node(feat=gain_dict["Index"],
                                      feat_type=gain_dict["Type"],
                                      feat_values=values[0:3],
                                      child_1=c_algorithm(x_0, y_0, gain_dict[0], node_dict),  # Recursive call
                                      child_2=c_algorithm(x_1, y_1, gain_dict[1], node_dict),
                                      child_3= c_algorithm(x_2, y_2, gain_dict[2], node_dict))
                return node_dict[key]  # Return the node we have just created
        return None


def column_unused(value, node_dict):
    # Search the node dictionary to check if a given column has been used yet in that branch of the decision tree
    # Input: value = index to be checked, node_dict = the node dictionary corresponding to that branch
    # Return: verdict = boolean True if the column has not been used yet, False otherwise
    verdict = True
    for key in node_dict:
        if isinstance(node_dict[key], str):  # If it is =='empty'
            return True
        elif isinstance(node_dict[key], int):  # Now we must check if the index in the dictionary == inputted index
            if node_dict[key] == value:
                verdict = False
                return verdict
            else:
                pass
        elif node_dict[key].feat == value:  # Ask this node if it using the inputted index
            verdict = False
    return verdict


def max_all_cols(x):
    # Find the max value across all columns and rows of an array
    # Input: x = an array of features
    # Return: maximum = the max value
    maximum = 0
    for i in range(x.shape[1]):
        if max(x[:, i]) > maximum:
            maximum = max(x[:, i])
    return maximum


def entropy(vector, value, y, y_classes):
    # Compute entropy for some value of an attribute
    # Input: vector = feature, value = some value in feature, y = some class vector, y_classes = list of classes in y
    # Return: Entropy
    e = 0
    for i in range(len(y_classes)):
        p = prob(vector, value, y, y_classes[i])
        if p == 0:
            pass
        else:
            e -= p * np.log2(p)
    return e


def prob(vector, value, y, y_value):
    # Probability of selecting a given value with a given class from a vector
    # Input: vector = feature, value = some value in feature, y = some class vector, y_value = target class
    # Return: Probability, a float
    # Prob(no. of instances with a given value & class / no. of instances with a given value)
    total = 0
    for i in range(len(vector)):
        if vector[i] == value and y[i] == y_value:
            total += 1
    # print("prob")
    return total / count(vector, value)


def info_gain(vector, values, y, global_ent):
    # Find overall information gain of some attribute
    # Input: vector = feature, values = list of values within said feature, y = class vector, global_ent = parent
    # node's entropy
    # Return: Dictionary containing the information gain, in key "Gain"
    gain = global_ent
    gain_dict = {}
    y_classes = find_unique_classes(y)
    for i in range(len(values)):
        p = count(vector, values[i]) / len(vector)
        ent = entropy(vector, values[i], y, y_classes)
        gain -= p * ent
        gain_dict[values[i]] = ent
    gain_dict["Gain"] = gain
    return gain_dict


def split_info(vector, values):
    # Input: vector = one feature, values = list of values within said feature
    # Returns SplitInfo, a float
    split = 0
    for i in range(len(values)):
        p = count(vector, values[i]) / len(vector)
        if p == 0:
            # Do nothing
            pass
        else:
            split -= p * np.log2(p)
    return split


def gain_ratio(vector, y, global_ent):
    # For one discrete feature, class vector and parent entropy value, compute the gain ratio
    # Input: vector = one feature, y = class vector, global_ent = parent node's entropy
    # Return: Dictionary containing the gain ratio
    values = find_unique_classes(vector)
    split = split_info(vector, values)
    gain_dict = info_gain(vector, values, y, global_ent)
    gain_dict["Gain Ratio"] = gain_dict["Gain"] / split
    gain = gain_dict["Gain"]
    return gain_dict


def gain_ratio_continuous(vector, y, global_ent):
    # Computes the information gain ratio of some continuous feature, at each of it's indices
    # Input: vector = one feature, y = class vector, global_ent = parent node's entropy
    # Return: Dictionary containing the best gain ratio for this feature, along with the index used to find it

    # Some data parsing
    data_mat = np.column_stack((vector, y))
    data_mat_copy = np.array(data_mat[data_mat[:, 0].argsort()])
    data_mat = np.array(data_mat[data_mat[:, 0].argsort()])
    # Initialise a dictionary to store the current best gain ratio
    best_gain = {"Gain Ratio": 0}
    initial_split_vector(data_mat_copy[:, 0], 0)  # Split the copy vector
    for i in range(len(data_mat_copy[:, 0]) - 1):
        # We loop to len - 1 here, weird bug when values has only one value, entropy becomes inf
        data_mat_copy[i, 0] = 0  # Split the copy vector based on current index
        gain_dict = gain_ratio(data_mat_copy[:, 0], data_mat_copy[:, 1], global_ent)
        gain_dict["Threshold"] = data_mat[i, 0]
        if gain_dict["Gain Ratio"] > best_gain["Gain Ratio"]:
            for key in gain_dict:
                best_gain[key] = gain_dict[key]
    # End loop
    # print("Found best gain ratio: ", gain_dict["Gain Ratio"])  #Debug
    return best_gain


def initial_split_vector(vector, index):
    # Split some vector such that all attributes below or equal to a certain index become 0, all others become 1
    # Input: vector, index
    # Return: modified vector
    for j in range(len(vector)):
        if j <= index:
            vector[j] = 0
        else:
            vector[j] = 1
    return None


def set_values(vector, value):
    # Sets all values in vector to the value passed
    # Input: vector
    # Return: modified vector
    for i in range(len(vector)):
        vector[i] = value
    return vector


def multi_class_algorithm(x, y, specified_training=False, test_x=None, test_y=None):
    # This is a helper function for c_algorithm, here we will parse the data using a one-vs-one approach
    # Input: x = feature array, y = class vector, specified_training = boolean (This will be true if the user passes
    # their own test/training data as opposed to having it done randomly) default = False, test_x = test features (only
    # used if specified_training=True) test_y = test classes (only used if specified_training=True)
    # Return: predicts = array containing the actual class of test data along with each model's prediction, stats =
    # list containing some basic info which will be written to the output file
    stats_list = []
    if specified_training:  # Only if user specifies the training/test data
        train_x = x.copy()
        train_y = y.copy()
        stats_list.append(len(train_y))
        stats_list.append(len(test_y))
    else:
        training_size = int(len(y) * (2 / 3))  # Split the data into two-thirds training, one-third test
        print("Number of rows in training data: ", training_size)
        training_indices = random.sample(range(len(y)-1), training_size)  # Take a random sample of indices to use
        # Specify training data
        train_x = x[training_indices, :]
        train_y = y[training_indices]
        stats_list.append(training_size)
        # Specify test data
        test_x = np.delete(x, training_indices, axis=0)
        test_y = np.delete(y, training_indices)
        stats_list.append(len(test_y))
    # Populate the node dictionary with empty strings
    node_dict = {}
    for i in range(x.shape[1]*3):
        # Testing many times, I found that a factor of 3 will not result in any more nodes needed than the node
        # dictionary allows
        node_dict["Node{0}".format(i)] = "empty"
    y_classes = find_unique_classes(train_y)
    num_models = int((len(y_classes) * (len(y_classes)-1))/2)  # Calculate the number of models we need to create
    stats_list.append(num_models)
    print("Number of one vs one models: ", num_models)
    # We store our predictions in an array, where
    # first column = actual class
    # all columns in between correspond to a one vs one model prediction
    # last column = final predicted class
    predicts = np.empty((len(test_y), num_models+2), dtype='U25')
    predicts[:, 0] = test_y
    # Now we create all of the one-vs-one models
    for i in range(num_models):
        new_node_dict = copy_dict(node_dict)  # Copy the node dictionary, as each model needs its own seperate nodes
        class_dict = {0: y_classes[i], 1: y_classes[(i+1) % len(y_classes)]}
        print("Multi-Class Algorithm, comparing: ", class_dict)
        new_train_y, new_train_x = one_vs_one(train_x, train_y, y_classes[i], y_classes[(i+1) % len(y_classes)])
        global_ent = global_entropy(new_train_y)
        root_node = c_algorithm(new_train_x, new_train_y, global_ent, new_node_dict)  # Call the actual algorithm
        display_nodes(new_node_dict)
        # Now classify all tests, and store these in our predicts array
        for j in range(len(test_y)):
            # Take each instance of the test data, one at a time
            row_to_test = np.concatenate((test_x[j, :], test_y[j:j+1]))
            temp = root_node.classify(row_to_test)
            predicted_class = class_dict.get(temp)
            predicts[j, i+1] = predicted_class
    return predicts, stats_list


def set_one_or_zero(vector, value):
    # Transform a vector into binary data, where any instances equal to the input value are 1, 0 otherwise
    # Input: vector = some vector, value = some value
    # Return: modified vector
    for i in range(len(vector)):
        if vector[i] == value:
            vector[i] = 1
        else:
            vector[i] = 0
    return vector


def copy_dict(dict):
    # For some unknown reason the inbuilt .copy() method for dictionaries was not working
    # This short function serves the same purpose as dictionary.copy()
    # Input: Some dictionary
    # Return: A copy of that dictionary
    new_dict = {}
    for key in dict:
        new_dict[key] = dict[key]
    return new_dict


def one_vs_one(x, vector, class_1, class_2):
    # Remove all instances NOT of class_1 and class_2 in class vector and feature array
    # Also set the classes to 0,1, as we cannot operate on strings
    # Input: x = feature array, vector = class vector, class_1 and class_2 are the target classes in this case
    # Return: modified features and classes separately

    bad_indices = []  # Store indices where class != class_1 or class_2
    new_vector = vector.copy()  # New vector will be the modified vector to be returned
    new_features = x  # New features will be the modified features to be returned

    # Iterate over the entire class vector
    for i in range(len(new_vector)):
        if new_vector[i] == class_1:
            new_vector[i] = 0
        elif new_vector[i] == class_2:
            new_vector[i] = 1
        else:
            bad_indices.append(i)
    new_vector = np.delete(new_vector, bad_indices)
    new_features = np.delete(x, bad_indices, axis=0)
    return new_vector, new_features


def global_entropy(y):
    # Calculate the global entropy, to be used when creating the root node
    # Input: The class vector
    # Returns: global_ent, a float containing the global entropy
    classes = find_unique_classes(y)
    global_ent = 0
    for i in range(len(classes)):
        p = count(y, classes[i]) / len(y)
        global_ent -= p * np.log2(p)
    # print("Global Entropy: ", global_ent)  # Used in DEBUG
    return global_ent


def display_nodes(node_dict):
    # Helper function to display all the nodes for a given iteration of the algorithm
    # Uses the class function show to achieve this
    # Input: A dictionary containing nodes
    # Return: None
    for key in node_dict:
        print(" ")  # To make the output legible
        print("Showing... ", node_dict[key])
        # If this key is empty (as empty is stored as a string)
        if isinstance(node_dict[key], str):
            pass  # Do nothing
        else:
            node_dict[key].show()  # Show the node
    return None


def find_best_prediction(predicts):
    # Appends the final prediction for each instance of the test data, to the prediction array
    # Input: The prediction array
    # Return: The now completed prediction array
    classes = find_unique_classes(predicts[:, 0])
    # Iterate over each row
    for j in range(len(predicts[:, 0])):
        class_dict = {}  # Storing a total for each predicted class
        # Populate the keys with each class
        # Set each total to 0
        for k in range(len(classes)):
            class_dict[classes[k]] = 0
        # Iterate over each column, excluding the first and last columns
        for i in range(1, predicts.shape[1] - 1):
            # Now increase the running total for that class
            for key in class_dict:
                if predicts[j, i] == key:
                    class_dict[key] += 1
        # Find the class which was predicted the most, in the one-vs-one models
        highest_total = 0
        highest_key = ""
        for key in class_dict:
            if class_dict[key] > highest_total:
                highest_total = class_dict[key]
                highest_key = key
        predicts[j, predicts.shape[1]-1] = highest_key
    return predicts


def accuracy(predicts):
    # Computes the prediction accuracy of a particular iteration of the algorithm
    # Input: predicts= a numpy array containing the actual classes and predicted class, for each model
    # Return: a float == prediction accuracy
    success = 0
    for i in range(len(predicts[:, 0])):
        if predicts[i, 0] == predicts[i, predicts.shape[1]-1]:
            success += 1
    return success/len(predicts[:, 0])


def write_array_to_file(predicts, accuracy, filename):
    # Write the prediction array to a specified file
    # Input: predicts= a numpy array containing the actual classes and predicted class, for each model, accuracy = the
    # prediction accuracy for this iteration, filename = a filename to write to
    # Return: None
    f = open(filename, 'a')
    np.savetxt(f, predicts, delimiter=", ", fmt="%s")
    f.write('Prediction accuracy: ' + '{}'.format(accuracy) + '\n')
    for i in range(3):
        f.write('\n')
    f.close()
    return None


def write_description(stats, avg_accuracy, filename):
    # Write the description of the algorithm as a whole to the end of the output file
    # Input: stats = A list of statistics about the algorithm, avg_accuracy = the average accuracy over 10 iterations,
    # filename = a filename to write to
    # Return: None
    f = open(filename, 'a')
    for i in range(2):
        f.write('\n')
    f.write('Results from the C4.5 decision tree algorithm are above.\n')
    f.write('The algorithm was run a total of ten times, and the average accuracy is reported below\n')
    f.write('The training data consisted of ' + '{}'.format(stats[0]) + ' instances,\n')
    f.write('While the test data consisted of ' + '{}'.format(stats[1]) + ' instances.\n')
    f.write('\n')
    f.write('A one to one multi-class approach was used for the data, \n')
    f.write('   where ' + '{}'.format(stats[2]) + ' models were created.\n')
    f.write('Each model is created using only two classes from the entire class list,\n')
    f.write('These models then vote on the final class, when predicting the test data.\n')
    f.write('\n')
    f.write('Above is the matrix containing the final class predictions, \n')
    f.write('   The leftmost column contains the original class of each instance of the test data,\n')
    f.write('   The rightmost column contains the final prediction of the algorithm,\n')
    f.write('   Each of the columns in between corresponds to the output from one model.\n')
    f.write('\n')
    f.write('The average prediction accuracy over ten executions was: ' + '{}'.format(avg_accuracy) + '\n')
    f.close()
    return None


# The node class, used to create the decision tree
# There is only two functions here besides init(), one to classify test instances
# and one to display info about the object
class Node:

    def __init__(self, feat=None, feat_type=None, threshold=None, feat_values=None, child_1=None, child_2=None, child_3=None, value=None):
        self.feat = feat  # The index of the original dataset that corresponds to this node
        self.feat_type = feat_type  # Either continuous or discrete
        self.threshold = threshold  # If self.feat_type = continuous, store the threshold value
        self.feat_values = feat_values # If self.feat_type = discrete, store values
        self.child_1 = child_1  # The child for values below//equal to the threshold
        self.child_2 = child_2  # The child for values above the threshold
        self.child_3 = child_3  # If a discrete feature has three values
        self.value = value  # If this is a leaf node, its associated class

    def classify(self, vector):
        if self.value is not None:
            return self.value  # If this is a leaf node, return the class
        else:
            temp_value = 0
            if self.feat_type == "Continuous":  # If continuous
                if vector[self.feat] <= self.threshold:  # If test instances less than / equal to threshold
                    temp_value = self.child_1.classify(vector)  # Pass test instances to child
                    # print("Passing value to child 1") DEBUG
                else:  # If test instances greater than threshold
                    temp_value = self.child_2.classify(vector)  # Pass test instance to child
                    # print("Passing value to child 2") DEBUG
            else:
                temp_value = 0
                if vector[self.feat] == self.feat_values[0]:
                    temp_value = self.child_1.classify(vector)
                elif vector[self.feat] == self.feat_values[1]:
                    temp_value = self.child_2.classify(vector)
                elif vector[self.feat] == self.feat_values[2]:
                    temp_value = self.child_3.classify(vector)
            # The child will pass the test instance to its children appropriately, eventually we reach a leaf node
            # The value of this node is passed back up to the root node, and returned here...
            return temp_value

    def show(self):
        # Print the attributes associated with this node
        print("Feature Index: ", self.feat)
        print("Feature type: ", self.feat_type)
        print("Threshold: ", self.threshold)
        print("Child1: ", self.child_1)
        print("Child2: ", self.child_2)
        print("Child3: ", self.child_3)
        print("Value: ", self.value)
        return None


# Read data from file
print("Please input the filename with path, including the extension.")
print("If the file is in the same directory as I am, no need to input path.")
print("I am assuming that the file is separated by tabs, not commas.")
print("If you wish to input training data and test data separately, please input the training data here")
filename = input()

# filename = "beer.txt"

data = pd.read_csv(filename, sep='\t', header=None)  # Read the data using pandas

print("Do you wish to input your own test/training data?")
print("If so, you will need to input them in seperate files(unfortunately)")
print("Please type 'yes' or 'no' ")
own_training_data = input()

# The following code contains a large amount of duplication, and will only be run in the event that a user wishes to
# input theeir own test/training data seperately
if own_training_data == 'yes':
    print("Please input the column index which holds the classes in your training data.")
    print("Note, I start counting at 0.")
    classcol_train = int(input())
    print("Please input the filename with path, including the extension, that holds the test data.")
    print("If the file is in the same directory as I am, no need to input path.")
    print("I am assuming that the file is separated by tabs, not commas.")
    test_data_filename = input()
    print("Please input the column index which holds the classes in your test data.")
    print("Note, I start counting at 0.")
    classcol_test = int(input())
    test_data = pd.read_csv(test_data_filename, sep='\t', header=None)  # Read the data using pandas
    train_Y = np.array(data.iloc[:, classcol_train])  # store column containing classes as Y
    train_X = np.array(data.loc[:, data.columns != classcol_train])  # store column(s) containing features as X
    test_Y = np.array(test_data.iloc[:, classcol_test])  # store column containing classes as Y
    test_X = np.array(test_data.loc[:, data.columns != classcol_test])  # store column(s) containing features as X
    print("Please input a filename, including a .csv extension, to store the output in")
    print("Note: If this file already exists, it will now be deleted")
    output_filename = input()
    answer = False
    while not answer:
        print("WARNING: I am about to delete the contents of the file you specified, if it exists")
        print("Type: 'yes' to proceed")
        if input() == 'yes':
            answer = True
    # A running total to store prediction accuracy over successive iterations
    total_accuracies = 0
    for i in range(10):
        predicts, stats = multi_class_algorithm(train_X, train_Y, own_training_data=True, test_X=test_X, test_Y=test_Y)
        predicts = find_best_prediction(predicts)
        current_accuracy = accuracy(predicts)
        total_accuracies += current_accuracy
        write_array_to_file(predicts, current_accuracy, output_filename)

    avg_accuracy = total_accuracies / 10
    write_description(stats, avg_accuracy, output_filename)
else:
    print("Please input the column index which holds the classes.")
    print("Note, I start counting at 0.")
    classcol = int(input())

    Y = np.array(data.iloc[:, classcol])  # store column containing classes as Y
    X = np.array(data.loc[:, data.columns != classcol])  # store column(s) containing features as X

    print("Please input a filename, including a .csv extension, to store the output in")
    print("I recommend 'output.csv' ")
    print("Note: If this file already exists, it will now be deleted")

    output_filename = input()

    answer = False
    while not answer:
        print("WARNING: I am about to delete the contents of the file you specified, if it exists")
        print("Type: 'yes' to proceed")
        if input() == 'yes':
            answer = True

    # We are appending to the file later on, thus we ensure its empty now
    try:
        os.remove(output_filename)
    except OSError:
        pass

    # A running total to store prediction accuracy over successive iterations
    total_accuracies = 0
    for i in range(10):
        predicts, stats = multi_class_algorithm(X, Y)
        predicts = find_best_prediction(predicts)
        current_accuracy = accuracy(predicts)
        total_accuracies += current_accuracy
        write_array_to_file(predicts, current_accuracy, output_filename)

    avg_accuracy = total_accuracies/10
    write_description(stats, avg_accuracy, output_filename)
