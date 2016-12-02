#!/usr/bin/env python

'''
Naive Bayes
Formulation: We formulate the dependence of mail type(spam/not spam) on the words using Naive Bayes Assumption.
            The assumption here is given the type of email the words are independant of each other

            P( S=1/W1,W2,...,Wn) is proportional to P( W1,W2,...,Wn/S=1) P(S=1)
            By Naive Bayes assumption,
            P( S=1/W1,W2,...,Wn) is proportional to P( W1/S=1) P( W2/S=1) P( W3/S=1) ...P( Wn/S=1)P( S=1)

            In the above equation we learn each of the terms using the training dataset provided

Program Working: The part of code which parses the email is common between train and test mode.
                 We initialize the model (Model class with model parameters) based on the technique selected
                 Based on the mode, we call either
                    train - which will simply count the prior occurences and liklihood occurences
                            After all emails training sets are processed we convert these counts to costs
                            These costs are saved to model file
                    test - During testing phase we load the costs from file to the model class variables
                         - Each document we calculate the negative log of posterior (in above formulation)
                         - Each document is classified as that class which has minimum negative log lilkihood
                         - We compare our predictions with ground truth and calculate the confusion matrix
Output:
    For Count based features:
        True Positive:1184
        True Negative:1382
        False Positive:16
        False Negative:104

    For Binary features:
        True Positive:1184
        True Negative:1382
        False Positive:16
        False Negative:104

'''


import sys
import os
import Model
import EmailParser


MODES = {"train", "test"}
TECHNIQUES = {"bayes", "dt"}

try:
    mode = sys.argv[1]
    tech = sys.argv[2]
    directory = sys.argv[3]
    model_path = sys.argv[4]
except IndexError:
    print("Usage: python spam.py mode technique dataset-directory model-file")
    sys.exit(1)

if mode not in MODES:
    print("Invalid mode. Allowed values: train, test")
    sys.exit(2)
if tech not in TECHNIQUES:
    print("Invalid technique. Allowed values: bayes (for naive bayes), dt (for decision tree)")
    sys.exit(3)

if not os.path.isdir(directory):
    print("Directory " + directory + " does not exist")
    sys.exit(4)
else:
    if not os.path.isdir(directory + "/spam/"):
        print("Directory " + directory + "/spam/" + " does not exist")
        sys.exit(5)
    if not os.path.isdir(directory + "/notspam/"):
        print("Directory " + directory + "/notspam/" + " does not exist")
        sys.exit(6)

p = EmailParser.Parser()
spam_email_texts = p.parse(directory + "/spam/")
non_spam_email_texts = p.parse(directory + "/notspam/")

# sys.exit(0)
if mode == "train":
    model = Model.Model(tech)
    model.train(spam_email_texts, non_spam_email_texts)
    model.save(model_path, tech)
    print(model)

elif mode == "test":
    model = Model.Model()
    model.load(model_path, tech)
    print(model)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for text in spam_email_texts:
        prediction = model.test(text)
        if prediction == "spam":
            true_positive += 1
        else:
            false_negative += 1
    for text in non_spam_email_texts:
        prediction = model.test(text)
        if prediction == "notspam":
            true_negative += 1
        else:
            false_positive += 1
    print len(spam_email_texts)
    print len(non_spam_email_texts)
    print("True Positive:" + str(true_positive))
    print("True Negative:" + str(true_negative))
    print("False Positive:" + str(false_positive))
    print("False Negative:" + str(false_negative))

'''
	1)Formulation
		->We considered each word in a mail as an attribute and each row as a mail.
		->In the decision tree each node has root element as word and number of occurance(1 in case of binary).It has left and right which point to another decision tree node 			or leaf "spam" or "notsapm"		 
	2)Working
		->We iterate over each mail and extract the words and return a list of words for each mail. We construct a dictionary with two keys(spam and notspam). Each item in dictionary is a list of mail. Each list will hold word and number of time it occurs. We follow the below step in recursion.
			a)We create set of words in both spam and notspam list. 
			b)calculate the entropy for each attribute(based on whether continous or binary)
			c) split the list based on word occurance(number of occurance in case of continous) in the mail or not.
			d) proceed to step (a)  again till the list has only spam or nonspam mails.
	 	->Testing start from root node. If the root word exist (number of occurance in case of continous) and move to left tree or right based on it. This will continue till spam or notspam is reached.
	3)Assumption and design decision.
	-	>We have used pickle package for writing and reading the decision tree. 
	4)Result
		->Binary
			True Positive:1207
			True Negative:1270
			False Positive:128
			False Negative:81

		->Continous
			True Positive:1227
			True Negative:1241
			False Positive:157
			False Negative:61

	

'''
