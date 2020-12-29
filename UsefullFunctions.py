import re
from gensim.models import KeyedVectors


def extract_from_twit(str):
    result = re.findall(r"@(\w+)", str)
    return result


# str1 = "I love @stackoverflow because #people are very #helpful!"
# print(extract_from_twit(str1))

with open('positive_word.txt', 'rb') as f:
    lines = f.readlines()
    print(lines[3:7])
