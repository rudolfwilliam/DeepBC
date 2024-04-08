"""*wrong* meta data for morphomnist dataset"""

vars = ['thickness', 'intensity', 'image']

attrs = ['thickness', 'intensity'] 

attr2int = {'intensity' : 0, 'thickness' : 1, 'image' : 2}

# order of the keys is important!
wrong_graph_structure = {'intensity' : [],
                         'thickness' : ['intensity'],
                         'image'     : ['intensity', 'thickness']}