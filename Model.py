import math
import operator
import pickle
import re
import sys

import IncorrectModelFileException
import ModelNotEmptyException
from DecisionTree import DecisionTree
from collections import Counter

sys.setrecursionlimit(99999)


class Model:
    model_type = None
    model_tree = None
    priors = {"spam": 0, "notspam": 0}
    prior_costs = {"spam": 0, "notspam": 0}
    likelihood_counts = {"spam": {}, "notspam": {}}
    likelihood_costs = {"spam": {}, "notspam": {}}
    minimum_ll = {"spam": 0.0000000001, "notspam": 0.0000000001}
    DeTr = {"spam": [], "notspam": []}
    words_list = set()

    def __init__(self, model_type=None):
        self.model_type = model_type

    def calculateEntropy(self,word,DT):
        word_exist_l = []
        word_nexist = 0.0
        word_spam_exist_l = []
        word_spam_nexist = 0.0
        word_nspam_exist_l = []
        word_nspam_nexist = 0.0
        tot_len = float(len(DT["spam"]) + len(DT["notspam"]))
        for spam_mail in DT["spam"]:
            if word in spam_mail:
                word_exist_l.append(spam_mail[word])
                word_spam_exist_l.append(spam_mail[word])
            else:
                word_nexist+=1
                word_spam_nexist+=1
        for non_spam_mail in DT["notspam"]:
            if word in non_spam_mail:
                word_exist_l.append(non_spam_mail[word])
                word_nspam_exist_l.append(non_spam_mail[word])
            else:
                word_nexist+=1
                word_nspam_nexist+=1
        
        
        word_exist = float(len(word_exist_l))
        word_spam_exist = float(len(word_spam_exist_l))
        word_nspam_exist = float(len(word_nspam_exist_l))
        if word_spam_exist == 0:
            entp_val_spam = 0.0
        else:
            entp_val_spam = (word_exist / tot_len) * \
            (-1 * (word_spam_exist / word_exist) * math.log(word_spam_exist / float(word_exist)))

        if word_nspam_exist == 0:
            entp_val_nspam = 0.0
        else:
            entp_val_nspam = (word_exist / tot_len) * \
            (-1 * (word_nspam_exist / word_exist) * math.log(word_nspam_exist / float(word_exist)))

        if word_spam_nexist == 0:
            entp_nval_spam = 0.0
        else:
            entp_nval_spam = (word_nexist / tot_len) * \
            (-1 * (word_spam_nexist / word_nexist) * math.log(word_spam_nexist / float(word_nexist)))

        if word_nspam_nexist == 0:
            entp_nval_nspam = 0.0
        else:
            entp_nval_nspam = (word_nexist / tot_len) * \
            (-1 * (word_nspam_nexist / word_nexist) * math.log(word_nspam_nexist / float(word_nexist)))
        result =  (word, [1,((entp_val_spam + entp_val_nspam) * (entp_nval_spam + entp_nval_nspam))])
        return result
            
    def buildDecisionTree(self, DT):
        entrList = {}
        if len(DT["spam"]) == 0:
            return "notspam"
        if len(DT["notspam"]) == 0:
            return "spam"
        for word in self.words_list:
            entrList[word] = self.calculateEntropy(word,DT)
        #print entrList
        min_value = max(entrList.values(), key=operator.itemgetter(1))
        #print min_value[0]
        self.words_list.remove(min_value[0])
        #print min_value
        left_tree = {"spam": [], "notspam": []}
        right_tree = {"spam": [], "notspam": []}
        for spam_mail in DT["spam"]:
            if min_value[0] in spam_mail:
                left_tree["spam"].append(spam_mail)
            else:
                right_tree["spam"].append(spam_mail)
        for non_spam_mail in DT["notspam"]:
            if min_value[0] in non_spam_mail:
                left_tree["notspam"].append(non_spam_mail)
            else:
                right_tree["notspam"].append(non_spam_mail)
        lt = self.buildDecisionTree(left_tree)
        rt = self.buildDecisionTree(right_tree)
        return DecisionTree((min_value[0],min_value[1][0]), lt, rt)

    def train(self, *args, **kwds):
        if len(args) == 2:
            words = args[0]
            spam_type = args[1]
            self.priors[spam_type] += 1
            # words = re.split(' |:|\.|,|\n', text)
            size = len(words)
            for word in words:
                if word in self.likelihood_counts[spam_type]:
                    self.likelihood_counts[spam_type][word] += 1
                else:
                    self.likelihood_counts[spam_type][word] = 1
            if self.model_type == "dt":
                # Do additional work here to create thresholds etc for decision tree model
                words = args[0]
                spam_type = args[1]
                self.priors[spam_type] += 1
                #words = re.split(' |:|\.|,|\n', text)
                word_count = dict(Counter(words))
                self.DeTr[spam_type].append(word_count)
                #print self.DeTr
                self.words_list.update(words)

        elif len(args) == 1:
            # May be we need this function to train spam/non-spam text as whole instead row wise
            whole_text = args[0]
            pass

    def calculate_probabilities(self):
        total_counts = sum(self.priors.itervalues())
        for prior, count in self.priors.iteritems():
            self.prior_costs[prior] = math.log(1 / ((1.0 * count) / total_counts))

        for prior, word_counts in self.likelihood_counts.iteritems():
            current_count = sum(word_counts.itervalues())
            for word, count in word_counts.iteritems():
                self.likelihood_costs[prior][word] = math.log(1 / ((1.0 * count) / current_count))

    def build_dt(self):
        return self.buildDecisionTree(self.DeTr)

    def is_spam_or_not(self, words_list):
        tempTree = self.model_tree
        while isinstance(tempTree, DecisionTree):
            if tempTree.root[0] in words_list and words_list.count(tempTree.root[0])>=tempTree.root[1]:
                tempTree = tempTree.left
            else:
                tempTree = tempTree.right
        return tempTree

    # Return the odds ratio of spam log likelihood probability v non spam log likelihood
    def test(self, words):
        if self.model_type == "bayes":
            spam_total_counts = sum(self.likelihood_counts["spam"].itervalues()) + 0.1
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            spam_prior = self.prior_costs["spam"]
            spam_result = 0
            # for word in re.split(' |:|\.|,|\n', text):
            for word in words:
                curr_cost = math.log(1 / (0.1 / spam_total_counts))
                if word in self.likelihood_costs["spam"]:
                    curr_cost = self.likelihood_costs["spam"][word]
                spam_result += curr_cost
            spam_result += spam_prior

            not_spam_total_counts = sum(self.likelihood_counts["notspam"].itervalues()) + 0.1
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            not_spam_prior = self.prior_costs["notspam"]
            notspam_result = 0
            # for word in re.split(' |:|\.|,|\n', text):
            for word in words:
                curr_cost = math.log(1 / (0.1 / not_spam_total_counts))
                if word in self.likelihood_costs["notspam"]:
                    curr_cost = self.likelihood_costs["notspam"][word]
                notspam_result += curr_cost
            notspam_result += not_spam_prior
            return spam_result - notspam_result
        else:
            # words_in_mail = re.split(' |:|\.|,|\n', text)
            result = self.is_spam_or_not(words)
            return result

    def save(self, file_path, tech, tree=None):
        print(file_path)
        if tech == "bayes":
            with open(file_path, "w") as fp:
                fp.write("Model:" + self.model_type + "\n")
                fp.write("Priors:" + str(len(self.prior_costs)) + "\n")
                for prior, count in self.prior_costs.iteritems():
                    fp.write(prior + ":" + str(count) + "\n")
                fp.write(
                    "LL:" + str(len(self.likelihood_costs["spam"]) + len(self.likelihood_costs["notspam"])) + "\n")
                for prior, word_counts in self.likelihood_costs.iteritems():
                    for word, count in word_counts.iteritems():
                        fp.write(prior + ":" + word + ":" + str(count) + "\n")
        if tech == "dt":
            with open(file_path, "w") as fp:
                '''
                fringe = []
                outputs = []
                fringe.append(tree)
                while len(fringe)>0:
                    element = fringe.pop()
                    outputs.append(element.root)
                    if isinstance(element.left,DecisionTree):
                        fringe.insert(0,element.left)
                    else:
                        outputs.append(element.left)
                    if isinstance(element.right,DecisionTree):
                        fringe.insert(0,element.right)
                    else:
                        outputs.append(element.right)
                fp.write(str(outputs))
                '''
                fp.write(pickle.dumps(tree))
                fp.close()

    def load(self, file_path, technique):
        if self.model_type is not None:
            raise ModelNotEmptyException.ModelNotEmptyException
        else:
            if technique == "bayes":
                with open(file_path, "r") as fp:
                    content = [x.strip('\n') for x in fp.readlines()]
                model_row = content.pop(0)
                if technique != model_row.split(":")[1]:
                    raise IncorrectModelFileException
                self.model_type = model_row.split(":")[1]
                while True:
                    next_row = content.pop(0)
                    if "LL" in next_row:
                        break
                    split_value = next_row.split(":")
                    self.priors[split_value[0]] = float(split_value[1])
                for likelihoods in content:
                    split_value = likelihoods.split(":")
                    self.likelihood_counts[split_value[0]][split_value[1]] = float(split_value[2])
            if technique == "dt":
                if self.model_type is not None:
                    raise ModelNotEmptyException.ModelNotEmptyException
                else:
                    with open(file_path, "r") as fp:
                        self.model_tree = pickle.loads(fp.read())
                        # print self.model_tree.left
                        fp.close()

    def __str__(self):
        ret_string = ""
        if self.model_type == "bayes":
            ret_string += "Top 10 words associated with spam (with strength of association):\n"
            ret_string += str(
                sorted(self.likelihood_costs["spam"].iteritems(), key=operator.itemgetter(1))[:10]) + "\n"
            ret_string += "Top 10 words associated with non spam (with strength of association):\n"
            ret_string += str(
                sorted(self.likelihood_costs["notspam"].iteritems(), key=operator.itemgetter(1))[:10]) + "\n"
            # if self.model_type == "dt":
            #    print self.words_list

        return ret_string
