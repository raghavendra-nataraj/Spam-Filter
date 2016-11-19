import email.parser
from os import listdir
from os.path import isfile, join
import string
import BeautifulSoup
import re


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True


class Parser:
    prsr = None

    def __init__(self):
        self.prsr = email.parser.Parser()

    def parse(self, folder_path):
        current_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for email_file in current_files:
            with open(folder_path + email_file, 'r') as fp:
                result = self.prsr.parse(fp)
                ctype = result.get_content_type()
                current_message = ""
                if "multipart" in ctype:
                    # for part in result.walk():
                    #     if part.get_content_type() == 'text/plain':
                    #         current_message += part.get_payload()
                    continue
                elif "html" in ctype:
                    current_message = result.get_payload()
                    soup = BeautifulSoup.BeautifulSoup(current_message)
                    texts = soup.findAll(text=True)
                    visible_texts = filter(visible, texts)
                    # print(visible_texts)
                    # text_message = html2text.html2text(current_message)
                    print("".join([c.decode("UTF-8") for c in visible_texts if
                                   c.decode("UTF-8") in string.letters or c.decode("UTF-8") in
                                   string.whitespace]))
                break
