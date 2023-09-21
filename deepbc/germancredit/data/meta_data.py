attrs = ['thickness', 'intensity'] 

attr2int = {'thickness' : 0, 'intensity' : 1, 'image' : 2}

graph_structure = {
                   'gender' : [],
                   'age' : [],
                   'credit' : ['gender', 'age'],
                   'duration' : ['credit']
                   }