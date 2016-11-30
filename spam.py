import sys
import os
import Model
import EmailParser
import pprint

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
