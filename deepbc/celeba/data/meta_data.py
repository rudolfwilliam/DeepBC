"""Meta data for celeba dataset. Order of attributes is important!"""

attrs = ['age', 'gender', 'bald', 'beard']
vars = ['age', 'gender', 'bald', 'beard', 'image']
 
# attribute order in celebA
attr2int = {'age' : 39, 'gender' : 20, 'bald' : 4, 'beard' : 24}

# order in continuous attributes celebA
vars2int = {'age' : 0, 'gender' : 1, 'bald' : 2, 'beard' : 3, 'image' : 4}

# see https://arxiv.org/pdf/2004.08697.pdf (group 1)
graph_structure = {'age'    : [],
                   'gender' : [],
                   'bald'   : ['age', 'gender'],
                   'beard'  : ['age', 'gender'],
                   'image'  : ['age', 'gender', 'bald', 'beard']}

# to show how graph affects results
wrong_graph_structure = {'age' : [],
                         }
