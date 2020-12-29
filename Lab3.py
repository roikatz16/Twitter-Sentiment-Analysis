from bs4 import BeautifulSoup
import urllib.request
import re
import sklearn as skp
import seaborn
import folium
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

data_frame = pd.read_csv("Train.csv")
print(data_frame.head())
