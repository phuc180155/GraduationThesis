data = ['econ_', 'bn_', 'email_']
dels = ['del_nodes', 'del_edges']
# with open('results','w') as r:
for i in data:
    for d in dels:
        path = i+d
        with open(path,'w') as f:
            print('ok')