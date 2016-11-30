import email.parser
import re
from os import listdir
from os.path import isfile, join
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from stemming.porter2 import stem
import string


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode("UTF-8"))):
        return False
    return True


class Parser:
    prsr = None
    stops = None
    h = None
    printable = None

    def __init__(self):
        self.prsr = email.parser.Parser()
        self.stops = set([word.encode("UTF-8") for word in stopwords.words('english')])

        self.h = HTMLParser()
        self.printable = set(string.printable)

    def html_handler(self, html_string):

        soup = BeautifulSoup(html_string)
        # kill all script and style elements
        # for script in soup(["script", "style"]):
        #    script.extract()  # rip it out
        # texts = soup.findAll(text=True)
        # get text
        # text = soup.get_text()
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # visible_texts = filter(visible, texts)
        string_texts = "".join([c.encode("UTF-8").lower() for c in text])
        # print ("====")
        # print(string_texts )
        word_list = re.sub("[ ]+", " ", string_texts)

        word_list = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "",
                           word_list)
        # word_list=filter(lambda x: x in self.printable, word_list)
        # word_list=self.h.unescape(word_list.encode("UTF-8"))
        return_words = []
        word_list = ''.join([x for x in word_list if x in string.printable])

        for word in re.split('\\|!|@|#|\$|%|\^|&|\*|\)|\[|\]|\(|_|\+|=|-|~|;|:|\?|\"|\'|\.| |\n|>|<|\t|/|\||,|\}|\{|`',
                             word_list):
            if word not in self.stops:
                word = re.sub("[ ]+", " ", word)
                stemmed_word = stem(word)
                if len(word) > 0 and len(word) < 15 and not word.isdigit():
                    return_words.append(stemmed_word)
        return return_words

    def plain_handler(self, plain_text):
        plain_text = re.sub("[ ]+", " ", plain_text)
        return_words = []
        plain_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "",
                            plain_text)
        # print(plain_text)
        for word in re.split('\\|!|@|#|\$|%|\^|&|\*|\)|\[|\]|\(|_|\+|=|-|~|;|:|\?|\"|\'|\.| |\n|>|<|\t|/|,|\||\}|\{|`',
                             plain_text):
            word = word.lower()
            if word not in self.stops:
                word = re.sub("[ ]+", " ", word)
                stemmed_words = stem(word)
                if len(word) > 0 and len(word) < 15 and not word.isdigit():
                    return_words.append(stemmed_words)
        return return_words

    def parse(self, folder_path):
        current_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        email_texts = []
        results = []

        for email_file in current_files:
            with open(folder_path + email_file, 'r') as fp:
                results.append((self.prsr.parse(fp), email_file))
        while len(results) > 0:
            result, file = results.pop()
            ctype = result.get_content_type()
            current_message = ""
            if result.is_multipart():
                for parts in result.walk():
                    if not parts.is_multipart():
                        if "html" in parts.get_content_type():
                            # print("==================================" + file + "==================================")
                            current_message = parts.get_payload()
                            email_texts.append(self.html_handler(current_message))
                        elif "plain" in ctype:
                            text = parts.get_payload()
                            email_texts.append(self.plain_handler(text))
                            # filtered_words = [word.lower() for word in text.split() if word not in self.stops]
                            # stemmed_words = [stem(word) for word in filtered_words]
                            # email_texts.append(stemmed_words)
            elif "html" in ctype:
                # print("=================================="+file+"==================================")
                current_message = result.get_payload()
                email_texts.append(self.html_handler(current_message))
            elif "plain" in ctype:
                text = result.get_payload()
                email_texts.append(self.plain_handler(text))
                # email_texts.append(stemmed_words)
                # else:
                #    print ctype
        return email_texts
