import pickle as pkl
random_seed = 1234
ATmask = [1845,1893]
TSmask = ['F','T','F','T','A','G','O','R','A','B','F','T','F','F','T','F','T','T','F','F','T','F','T','F','T','T','A','A','G','O','R','F','T','F','T','T','F','T','F','T','F','T','F','T','O','T','F','F']

with open('/data/catle/Projects/macnetwork/data/gen_gqa_objects_answerVocabSemantic.pkl','rb') as fh:
    ASmask = pkl.load(fh)

def init():
    global numCW
    numCW = 0
