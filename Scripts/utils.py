def pre_processes_y(y):
    for i in range(len(y)):
        if(int(y[i]) > 1):
            y[i] = str(1)
    
    return y

def change(y):
    for i in range(len(y)):
        if(y[i] == 0):
            y[i] = 1
        else:
            y[i] = 2
            
    return y