import os
import re
from gensim.models import KeyedVectors
import math


from scipy.sparse import csr_matrix, vstack
A = csr_matrix([[1, 3, 9], [4, 0, 6]])
B = csr_matrix([[5, 6, 7]])
print(vstack([A, B]).toarray())

