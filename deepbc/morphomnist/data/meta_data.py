"""meta data for morphomnist dataset"""

vars = ['thickness', 'intensity', 'image']

attrs = ['thickness', 'intensity'] 

attr2int = {'thickness' : 0, 'intensity' : 1, 'image' : 2}

# order of the keys is important!
graph_structure = {'thickness' : [],
                   'intensity' : ['thickness'],
                   'image'     : ['intensity', 'thickness']}