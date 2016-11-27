import email.parser
import re
from os import listdir
from os.path import isfile, join
from HTMLParser import HTMLParser
import BeautifulSoup
from nltk.corpus import stopwords
from stemming.porter2 import stem


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True


class Parser:
    prsr = None
    stops = None
    h = None

    def __init__(self):
        self.prsr = email.parser.Parser()
        self.stops = set([word.encode("UTF-8") for word in stopwords.words('english')])

        self.h = HTMLParser()

    def html_handler(self, html_string):

        soup = BeautifulSoup.BeautifulSoup(html_string)
        # kill all script and style elements
        # for script in soup(["script", "style"]):
        #    script.extract()  # rip it out
        texts = soup.findAll(text=True)
        # get text
        # text = soup.get_text()
        visible_texts = filter(visible, texts)
        string_texts = "".join([c.encode("UTF-8").lower() for c in visible_texts])
        word_list = re.sub("[ ]+", " ", string_texts)
        word_list = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "",
                           word_list)
        return_words = []
        for word in re.split('\\|\!|@|#|\$|%|\^|&|\*|\)|\[|\]|\(|_|\+|=|-|~|;|:|\?|\"|\'|\.| |\n|>|<|\t|/|,',
                             word_list):
            if word not in self.stops:
                #               re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', word) is None:
                #                    if re.match('ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                #                               word) is None:
                stemmed_word = stem(word)
                return_words.append(stemmed_word)
        return return_words

    def plain_handler(self, plain_text):
        plain_text = re.sub("[ ]+", " ", plain_text)
        return_words = []
        plain_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "",
                            plain_text)
        for word in re.split('\\|\!|@|#|\$|%|\^|&|\*|\)|\[|\]|\(|_|\+|=|-|~|;|:|\?|\"|\'|\.| |\n|>|<|\t|/|,',
                             plain_text):
            word = word.lower()
            if word not in self.stops:
                word = re.sub("[ ]+", " ", word)
                #                if re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", word) is None:
                stemmed_words = stem(word)
                return_words.append(stemmed_words)
        return return_words

    def parse(self, folder_path):
        current_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        email_texts = []
        results = []

        for email_file in current_files:
            with open(folder_path + email_file, 'r') as fp:
                results.append(self.prsr.parse(fp))
        while len(results) > 0:
            result = results.pop()
            ctype = result.get_content_type()
            current_message = ""
            if result.is_multipart():
                for parts in result.walk():
                    if not parts.is_multipart():
                        if "html" in parts.get_content_type():
                            current_message = parts.get_payload()
                            email_texts.append(self.html_handler(current_message))
                        elif "plain" in ctype:
                            text = parts.get_payload()
                            email_texts.append(self.plain_handler(text))
                            # filtered_words = [word.lower() for word in text.split() if word not in self.stops]
                            # stemmed_words = [stem(word) for word in filtered_words]
                            # email_texts.append(stemmed_words)
            elif "html" in ctype:
                current_message = result.get_payload()
                email_texts.append(self.html_handler(current_message))
            elif "plain" in ctype:
                text = result.get_payload()
                email_texts.append(self.plain_handler(text))
                # email_texts.append(stemmed_words)
                # else:
                #    print ctype
        return email_texts
