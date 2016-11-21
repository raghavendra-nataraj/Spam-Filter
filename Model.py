import math
import pprint
import ModelNotEmptyException, IncorrectModelFileException
import operator
import re

class Model:
    model_type = None
    priors = {"spam": 0, "notspam": 0}
    likelihood_counts = {"spam": {}, "notspam": {}}
    minimum_ll = {"spam": 0.0000000001, "notspam": 0.0000000001}
    posterior_probabilites = {"spam": {}, "notspam": {}}

    def __init__(self, model_type=None):
        self.model_type = model_type

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
                pass

        elif len(args) == 1:
            # May be we need this function to train spam/non-spam text as whole instead row wise
            whole_text = args[0]
            pass

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

    def save(self, file_path):
        print(file_path)
        with open(file_path, "w") as fp:
            fp.write("Model:" + self.model_type + "\n")
            fp.write("Priors:" + str(len(self.priors)) + "\n")
            for prior, count in self.priors.iteritems():
                fp.write(prior + ":" + str(count) + "\n")
            fp.write("LL:" + str(len(self.likelihood_counts["spam"]) + len(self.likelihood_counts["notspam"])) + "\n")
            for prior, word_counts in self.likelihood_counts.iteritems():
                for word, count in word_counts.iteritems():
                    fp.write(prior + ":" + word + ":" + str(count) + "\n")

    def load(self, file_path, technique):
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

    def __str__(self):
    
        if self.model_type == "bayes":
            print("Top 10 words associated with spam (with strength of association):")
            pprint.pprint(dict(sorted(self.posterior_probabilites["spam"].iteritems(), key=operator.itemgetter(1),
                                      reverse=True)[:10]))
            print("Top 10 words associated with non spam (with strength of association):")
            pprint.pprint(dict(sorted(self.posterior_probabilites["notspam"].iteritems(), key=operator.itemgetter(1),
                                      reverse=True)[:10]))

