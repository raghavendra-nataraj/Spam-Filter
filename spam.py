import sys
import os

MODES = {"train", "test"}
TECHNIQUES = {"bayes", "dt"}
try:
    mode = sys.argv[1]
    tech = sys.argv[2]
    directory = sys.argv[3]
    model = sys.argv[4]
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
    if not os.path.isdir(directory + "/spam"):
        print("Directory" + directory + "/spam" + " does not exist")
        sys.exit(5)
    if not os.path.isdir(directory + "/notspam"):
        print("Directory" + directory + "/notspam" + " does not exist")
        sys.exit(6)

