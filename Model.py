import math


class Model:
    model_type = None
    priors = {"spam": 0, "notspam": 0}
    likelihood_counts = {"spam": {}, "notspam": {}}
    minimum_ll = {"spam": 0.0000000001, "notspam": 0.0000000001}

    def __init__(self, model_type="bayes"):
        self.model_type = model_type

    def train(self, *args, **kwds):
        if len(args) == 2:
            text = args[0]
            spam_type = args[1]
            self.priors[spam_type] += 1
            words = text.split()
            for word in words:
                if word in self.likelihood_counts[spam_type]:
                    self.likelihood_counts[spam_type][word] += 1
                else:
                    self.likelihood_counts[spam_type][word] = 1
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
            spam_total_counts = sum(self.likelihood_counts["spam"].itervalues())
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            spam_prior = self.priors["spam"]
            spam_result = 0
            for word in text.split():
                curr_prob = (1.0 * 0.1) / spam_total_counts
                if word in self.likelihood_counts["spam"]:
                    curr_prob = (1.0 * self.likelihood_counts["spam"][word]) / spam_total_counts
                spam_result += math.log(curr_prob)
            spam_result += math.log(spam_prior)

            notspam_total_counts = sum(self.likelihood_counts["notspam"].itervalues())
            # spam_min_value = min(self.likelihood_counts["spam"].itervalues())
            notspam_prior = self.priors["notspam"]
            notspam_result = 0
            for word in text.split():
                curr_prob = (1.0 * 0.1) / notspam_total_counts
                if word in self.likelihood_counts["notspam"]:
                    curr_prob = (1.0 * self.likelihood_counts["notspam"][word]) / notspam_total_counts
                notspam_result += math.log(curr_prob)
            notspam_result += math.log(notspam_prior)

            return spam_result / notspam_result
