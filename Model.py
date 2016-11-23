import math
import pprint
import ModelNotEmptyException, IncorrectModelFileException
import operator
import pickle
import re
import sys
from DecisionTree import DecisionTree


sys.setrecursionlimit(9999)
class Model:
    model_type = None
    modet_tree = None
    priors = {"spam": 0, "notspam": 0}
    likelihood_counts = {"spam": {}, "notspam": {}}
    minimum_ll = {"spam": 0.0000000001, "notspam": 0.0000000001}
    posterior_probabilites = {"spam": {}, "notspam": {}}
    DeTr = {"spam":[],"notspam":[]}
    words_list = set()
    
    def __init__(self, model_type=None):
        self.model_type = model_type

    def calcEntropy(self,DT=DeTr):
        word_exist = 0
        word_nexist = 0
        word_spam_exist = 0
        word_spam_nexist = 0
        word_nspam_exist = 0
        word_nspam_nexist = 0
        entrList = {}
        if len(DT["spam"])==0:
            return "notspam"
        if len(DT["notspam"])==0:
            return "spam"
        tot_len = len(DT["spam"])+len(DT["notspam"])
        for word in self.words_list:
            if not word.isalpha():
                continue
            for spam_mail in DT["spam"]:
                if word in spam_mail:
                    word_exist+=1
                    word_spam_exist+=1
                else:
                    word_nexist+=1
                    word_spam_nexist+=1
            for non_spam_mail in DT["notspam"]:
                if word in non_spam_mail:
                    word_exist+=1
                    word_nspam_exist+=1
                else:
                    word_nexist+=1
                    word_nspam_nexist+=1
            
            if word_spam_exist==0:
                entp_val_spam = 0
            else:
                entp_val_spam = (word_exist/tot_len)*\
                (-1*(word_spam_exist/word_exist)*math.log(word_spam_exist/float(word_exist)))
                
            if word_nspam_exist==0:
                entp_val_nspam = 0
            else:
                entp_val_nspam = (word_exist/tot_len)*\
                (-1*(word_nspam_exist/word_exist)*math.log(word_nspam_exist/float(word_exist)))
                
            if word_spam_nexist==0:
                entp_nval_spam = 0
            else:
                entp_nval_spam = (word_nexist/tot_len)*\
                (-1*(word_spam_nexist/word_nexist)*math.log(word_spam_nexist/float(word_nexist)))

            if word_nspam_nexist==0:
                entp_nval_nspam = 0
            else:
                entp_nval_nspam = (word_nexist/tot_len)*\
                (-1*(word_nspam_nexist/word_nexist)*math.log(word_nspam_nexist/float(word_nexist)))
            entrList[word] = (word,((entp_val_spam+entp_val_nspam) * (entp_nval_spam+entp_nval_nspam)))
        min_value =  min(entrList.values(),key=operator.itemgetter(1))
        self.words_list.remove(min_value[0])
        left_tree =  {"spam":[],"notspam":[]}
        right_tree = {"spam":[],"notspam":[]}
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
        lt = self.calcEntropy(left_tree)
        rt = self.calcEntropy(right_tree)
        return DecisionTree(min_value[0],lt,rt)
        
    def train(self, *args, **kwds):
        if len(args) == 2:
            text = args[0]
            spam_type = args[1]
            self.priors[spam_type] += 1
            words = re.split(' |:|\.|,|\n',text)
            for word in words:
                if word.lower() in self.likelihood_counts[spam_type]:
                    self.likelihood_counts[spam_type][word.lower()] += 1
                else:
                    self.likelihood_counts[spam_type][word.lower()] = 1
            if self.model_type == "dt":
                # Do additional work here to create thresholds etc for decision tree model
                text = args[0]
                spam_type = args[1]
                self.priors[spam_type] += 1
                words = re.split(' |:|\.|,|\n',text)
                self.DeTr[spam_type].append(words)
                self.words_list.update(words)

        elif len(args) == 1:
            # May be we need this function to train spam/non-spam text as whole instead row wise
            whole_text = args[0]
            pass

    def build_dt(self):
        return self.calcEntropy()
            
        
    # Return the odds ratio of spam log likelihood probability v non spam log likelihood
    def test(self, text):
        if self.model_type == "bayes":
            spam_total_counts = sum(self.likelihood_counts["spam"].itervalues())+0.1
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            spam_prior = self.priors["spam"]
            spam_result = 0
            for word in re.split(' |:|\.|,|\n',text):
                curr_prob = (1.0 * 0.1) / spam_total_counts
                if word.lower() in self.likelihood_counts["spam"]:
                    curr_prob = (1.0 * self.likelihood_counts["spam"][word.lower()]) / spam_total_counts
                if word.lower() not in self.posterior_probabilites["spam"]:
                    self.posterior_probabilites["spam"][word.lower()] = curr_prob
                spam_result += math.log(curr_prob)
            spam_result += math.log(float(spam_prior))

            notspam_total_counts = sum(self.likelihood_counts["notspam"].itervalues())+0.1
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            notspam_prior = self.priors["notspam"]
            notspam_result = 0
            for word in text.split():
                curr_prob = (1.0 * 0.1) / notspam_total_counts
                if word.lower() in self.likelihood_counts["notspam"]:
                    curr_prob = (1.0 * self.likelihood_counts["notspam"][word.lower()]) / notspam_total_counts
                    if word.lower() not in self.posterior_probabilites["notspam"]:
                        self.posterior_probabilites["notspam"][word.lower()] = curr_prob
                notspam_result += math.log(curr_prob)
            notspam_result += math.log(float(notspam_prior))
            return spam_result / notspam_result

    def save(self, file_path,tech,tree=None):
        print(file_path)
        if tech == "bayes":
            with open(file_path, "w") as fp:
                fp.write("Model:" + self.model_type + "\n")
                fp.write("Priors:" + str(len(self.priors)) + "\n")
                for prior, count in self.priors.iteritems():
                    fp.write(prior + ":" + str(count) + "\n")
                fp.write("LL:" + str(len(self.likelihood_counts["spam"]) + len(self.likelihood_counts["notspam"])) + "\n")
                for prior, word_counts in self.likelihood_counts.iteritems():
                    for word, count in word_counts.iteritems():
                        fp.write(prior + ":" + word + ":" + str(count) + "\n")
        if tech=="dt":
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
        if technique =="bayes":
            if self.model_type is not None:
                raise ModelNotEmptyException.ModelNotEmptyException
            else:
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
                    self.priors[split_value[0]] = int(split_value[1])
                for likelihoods in content:
                    split_value = likelihoods.split(":")
                    self.likelihood_counts[split_value[0]][split_value[1]] = int(split_value[2])
        if technique =="dt":
            if self.model_type is not None:
                raise ModelNotEmptyException.ModelNotEmptyException
            else:
                with open(file_path, "r") as fp:
                    tree = pickle.loads(fp.read())
                    retrn tree
                        
                        
    

    def __str__(self):
    
        if self.model_type == "bayes":
            print("Top 10 words associated with spam (with strength of association):")
            pprint.pprint(dict(sorted(self.posterior_probabilites["spam"].iteritems(), key=operator.itemgetter(1),reverse=True)[:10]))
            print("Top 10 words associated with non spam (with strength of association):")
            pprint.pprint(dict(sorted(self.posterior_probabilites["notspam"].iteritems(), key=operator.itemgetter(1),reverse=True)[:10]))
        #if self.model_type == "dt":
        #    print self.words_list
            
