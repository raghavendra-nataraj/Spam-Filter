import sys
import os
import Model
import EmailParser
from os import listdir
from os.path import isfile, join

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
    print("Directory" + directory + " does not exist")
    sys.exit(4)
else:
    if not os.path.isdir(directory + "/spam/"):
        print("Directory" + directory + "/spam/" + " does not exist")
        sys.exit(5)
    if not os.path.isdir(directory + "/notspam/"):
        print("Directory" + directory + "/notspam/" + " does not exist")
        sys.exit(6)

p = EmailParser.Parser()
spam_email_texts = p.parse(directory + "/spam/")
non_spam_email_texts = p.parse(directory + "/notspam/")

if mode == "train":
    model = Model.Model(tech)
    for text in spam_email_texts:
        model.train(text, "spam")
    for text in non_spam_email_texts:
        model.train(text, "notspam")
    model.save(model_path)
    print(model)
elif mode == "test":
    model = Model.Model()
    model.load(model_path, tech)
    spam_ratios = []
    for text in spam_email_texts:
        spam_ratios.append(model.test(text))
    true_positive = (1.0 * sum(i > 1 for i in spam_ratios))
    true_negative = len(spam_ratios) - true_positive
    non_spam_ratios = []
    for text in spam_email_texts:
        non_spam_ratios.append(model.test(text))
    false_positive = (1.0 * sum(i > 1 for i in non_spam_ratios))
    false_negative = len(spam_ratios) - false_positive
    print(true_positive)
    print(true_negative)
    print(false_positive)
    print(false_negative)
