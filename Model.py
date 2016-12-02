import math
import operator
import pickle
import pprint
import sys

import IncorrectModelFileException
import ModelNotEmptyException
from DecisionTree import DecisionTree
from collections import Counter


class Model:
    model_type = None
    priors = {"spam": 0, "notspam": 0}
    prior_costs = {"spam": 0, "notspam": 0}
    likelihood_counts = {"spam": {}, "notspam": {}}
    likelihood_costs = {"spam": {}, "notspam": {}}
    minimum_ll = {"spam": 0.0000000001, "notspam": 0.0000000001}
    spam_total_counts = 0.1
    notspam_total_counts = 0.1
    model_tree = None
    DeTr = {"spam": [], "notspam": []}
    wordslist_g = set()
    splitwords = set()

    def __init__(self, model_type=None):
        self.model_type = model_type

    def print_tree(self, tree, level, ret_lst):
        ret_lst.append((str("-----" * level) + str(tree.root)))
        if level == 4:
            return ""
        if isinstance(tree.left, DecisionTree):
            self.print_tree(tree.left, level + 1, ret_lst)
        else:
            ret_lst.append((str("-----" * (level + 1)) + str(tree.left)))

        if isinstance(tree.right, DecisionTree):
            self.print_tree(tree.right, level + 1, ret_lst)
        else:
            ret_lst.append((str("-----" * (level + 1)) + str(tree.right)))

    def __str__(self):
        ret_string = ""
        if self.model_type == "bayes":
            ret_string += "Top 10 words associated with spam (with strength of association):\n"
            ret_string += pprint.pformat(str(
                sorted(self.likelihood_costs["spam"].iteritems(), key=operator.itemgetter(1))[:10])) + "\n"
            ret_string += "Top 10 words associated with non spam (with strength of association):\n"
            ret_string += pprint.pformat(
                str(sorted(self.likelihood_costs["notspam"].iteritems(), key=operator.itemgetter(1))[:10])) + "\n"
        elif self.model_type == "dt":
            ret_lst = []
            self.print_tree(self.model_tree, 0, ret_lst)
            ret_string = "\n".join(ret_lst)
        return ret_string

    def train(self, spam_list, non_spam_list):
        for text in spam_list:
            self.train_rows(text, "spam")
        for text in non_spam_list:
            self.train_rows(text, "notspam")

        if self.model_type == "bayes":
            self.calculate_probabilities()

        if self.model_type == "dt":
            print "tree start"
            self.model_tree = self.build_dt()
            print "Tree Over"

    def train_rows(self, words, spam_type):
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
            self.priors[spam_type] += 1
            # words = re.split(' |:|\.|,|\n', text)
            word_count = dict(Counter(words))
            self.DeTr[spam_type].append(word_count)
            # print self.DeTr
            self.wordslist_g.update(words)

    # Return the odds ratio of spam log likelihood probability v non spam log likelihood
    def test(self, words):
        if self.model_type == "bayes":

            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            spam_prior = self.prior_costs["spam"]
            spam_result = 0
            # for word in re.split(' |:|\.|,|\n', text):
            for word in words:
                curr_cost = math.log(1 / (0.1 / self.spam_total_counts))
                if word in self.likelihood_costs["spam"]:
                    curr_cost = self.likelihood_costs["spam"][word]

                spam_result += curr_cost
            spam_result += spam_prior

            not_spam_total_counts = self.notspam_total_counts
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
            return "spam" if spam_result - notspam_result < 0 else "notspam"
        else:
            # words_in_mail = re.split(' |:|\.|,|\n', text)
            result = self.is_spam_or_not(words)
            return result

    def save(self, file_path, tech):
        if tech == "bayes":
            with open(file_path, "w") as fp:
                fp.write("Model:" + self.model_type + "\n")
                fp.write("Priors:" + str(len(self.prior_costs)) + "\n")
                for prior, count in self.prior_costs.iteritems():
                    fp.write(prior + ":" + str(count) + "\n")
                fp.write("LL Counts:" + str(len(self.prior_costs)) + "\n")
                fp.write("Spam" + ":" + str(self.spam_total_counts) + "\n")
                fp.write("NotSpam" + ":" + str(self.notspam_total_counts) + "\n")
                fp.write(
                    "LL:" + str(len(self.likelihood_costs["spam"]) + len(self.likelihood_costs["notspam"])) + "\n")
                for prior, word_counts in self.likelihood_costs.iteritems():
                    for word, count in word_counts.iteritems():
                        fp.write(prior + ":" + word + ":" + str(count) + "\n")
        if tech == "dt":
            with open(file_path, "w") as fp:
                fp.write(pickle.dumps(self.model_tree))
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
                next_loop_count = 0
                while True:
                    next_row = content.pop(0)
                    if "LL" in next_row:
                        count_split = next_row.split(":")
                        next_loop_count = count_split[1]
                        break
                    split_value = next_row.split(":")
                    self.prior_costs[split_value[0]] = float(split_value[1])
                spam_count_row = content.pop(0)
                notspam_count_row = content.pop(0)
                self.spam_total_counts = float(spam_count_row.split(":")[1])
                self.notspam_total_counts = float(notspam_count_row.split(":")[1])
                content.pop(0)
                for likelihoods in content:
                    split_value = likelihoods.split(":")
                    word = split_value[1]
                    for index in range(2, len(split_value) - 1):
                        word += ":" + split_value[index]
                    cost = split_value[len(split_value) - 1]
                    if split_value[0] not in self.likelihood_costs:
                        self.likelihood_costs[split_value[0]] = {}
                    self.likelihood_costs[split_value[0]][word] = float(cost)
            if technique == "dt":
                if self.model_type is not None:
                    raise ModelNotEmptyException.ModelNotEmptyException
                else:
                    with open(file_path, "r") as fp:
                        self.model_tree = pickle.loads(fp.read())
                        # print self.model_tree.left


                        fp.close()

    def calculateEntropy(self, word, DT):
        word_exist_l = []
        result = []
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
                word_nexist += 1
                word_spam_nexist += 1
        for non_spam_mail in DT["notspam"]:
            if word in non_spam_mail:
                word_exist_l.append(non_spam_mail[word])
                word_nspam_exist_l.append(non_spam_mail[word])
            else:
                word_nexist += 1
                word_nspam_nexist += 1
        # possible issues
        binary = False
        if binary:
            word_iterList = [1]
        else:
            word_iterList = set(word_exist_l)
        for num in word_iterList:
            greater_s = [x for x in word_spam_exist_l if x >= num]
            lesser_s = [x for x in word_spam_exist_l if x < num]
            greater_ns = [x for x in word_nspam_exist_l if x >= num]
            lesser_ns = [x for x in word_nspam_exist_l if x < num]
            # Till here
            word_exist = float(len(greater_s) + len(greater_ns))
            word_spam_exist = float(len(greater_s))
            word_nspam_exist = float(len(greater_ns))
            word_nexist += float(len(lesser_s) + len(lesser_ns))
            word_spam_nexist += float(len(lesser_s))
            word_nspam_nexist += float(len(lesser_ns))
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
                                 (-1 * (word_spam_nexist / word_nexist) * math.log(
                                     word_spam_nexist / float(word_nexist)))

            if word_nspam_nexist == 0:
                entp_nval_nspam = 0.0
            else:
                entp_nval_nspam = (word_nexist / tot_len) * \
                                  (-1 * (word_nspam_nexist / word_nexist) * math.log(
                                      word_nspam_nexist / float(word_nexist)))
            result.append([num, ((entp_val_spam + entp_val_nspam) + (entp_nval_spam + entp_nval_nspam))])
        result_value = min(result, key=operator.itemgetter(1))
        return (word, result_value)

    def buildDecisionTree(self, DT, wordlist):
        entrList = {}
        if len(DT["spam"]) == 0:
            return "notspam"
        if len(DT["notspam"]) == 0:
            return "spam"
        for word in wordlist:
            entrList[word] = self.calculateEntropy(word, DT)
        # print entrList
        min_value = min(entrList.values(), key=operator.itemgetter(1))
        wordlist.discard(min_value[0])
        self.splitwords.add(min_value[0])
        left_tree = {"spam": [], "notspam": []}
        right_tree = {"spam": [], "notspam": []}
        wordlist_l = set()
        wordlist_r = set()
        for spam_mail in DT["spam"]:
            if len(spam_mail)>0:
                if min_value[0] in spam_mail and spam_mail[min_value[0]] >= min_value[1][0]:
                    wordlist_l.update(spam_mail.keys())
                    left_tree["spam"].append(spam_mail)
                else:
                    wordlist_r.update(spam_mail.keys())
                    right_tree["spam"].append(spam_mail)
        for non_spam_mail in DT["notspam"]:
            if len(non_spam_mail)>0:
                if min_value[0] in non_spam_mail and non_spam_mail[min_value[0]] >= min_value[1][0]:
                    wordlist_l.update(non_spam_mail.keys())
                    left_tree["notspam"].append(non_spam_mail)
                else:
                    wordlist_r.update(non_spam_mail.keys())
                    right_tree["notspam"].append(non_spam_mail)
        wordlist_l.difference_update(self.splitwords)
        wordlist_r.difference_update(self.splitwords)
        lt = self.buildDecisionTree(left_tree, wordlist_l)
        rt = self.buildDecisionTree(right_tree, wordlist_r)
        return DecisionTree((min_value[0], min_value[1][0]), lt, rt)

    def build_dt(self):
        return self.buildDecisionTree(self.DeTr, self.wordslist_g)

    def is_spam_or_not(self, words_list):
        tempTree = self.model_tree
        while isinstance(tempTree, DecisionTree):
            if tempTree.root[0] in words_list and words_list.count(tempTree.root[0]) >= tempTree.root[1]:
                tempTree = tempTree.left
            else:
                tempTree = tempTree.right
        return tempTree

    def calculate_probabilities(self):
        self.spam_total_counts = sum(self.likelihood_counts["spam"].itervalues()) + 0.1
        self.notspam_total_counts = sum(self.likelihood_counts["notspam"].itervalues()) + 0.1
        total_counts = sum(self.priors.itervalues())
        for prior, count in self.priors.iteritems():
            self.prior_costs[prior] = math.log(1 / ((1.0 * count) / total_counts))
        for prior, word_counts in self.likelihood_counts.iteritems():
            current_count = sum(word_counts.itervalues())
            for word, count in word_counts.iteritems():
                self.likelihood_costs[prior][word] = math.log(1 / ((1.0 * count) / current_count))
