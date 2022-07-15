import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
COLORS = list(colors.CSS4_COLORS.keys())
def plotClust2D(labels, M, axes = ['max','tmax'],COLORS =COLORS,colorsLst = None,legend = None):
    numColors = len(COLORS)
    #labels as provided by the labels_ from DBSCAN
    #M a tall matrix where each row is a coordinate triple
    #can handle up to seven clusters with the colors
    groups = {}
    labels = (labels).astype(int)
    plt.figure(figsize = (7, 7))
    numLabels = len(set(labels))
    leg =[]
    for k in range(numLabels):
        i = list(set(labels))[k]
        groups[i] = np.array([M[j] for j in range(M.shape[0]) if labels[j]==i],dtype='float16')
        leg+=[str(k)]
        if colorsLst == None:
            Color =  COLORS[-1 -numColors//numLabels*k]
        else:
            Color = colorsLst[k]
        plt.scatter(groups[i][:,0],groups[i][:,1],
                     color =Color)
    if legend==None:
        plt.legend(leg)
    else:
        plt.legend(legend)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.title(str(len(set(labels)))+' clusters' )
    plt.show()
    
def splitSeqs(seq, n_steps_in, n_steps_out):
    '''
    taking input 1D sequence {seq}, generates {seq - n_steps_in - n_steps_out +1} x, y pairings 
    of consecutive sequential data with the x having dimension {n_steps_in} and y having dimension
    {n_steps_out}
    '''
    X, Y = [], []
    i = 0
    while i+n_steps_in+n_steps_out<=len(seq):
        x,y = seq[i:i+n_steps_in], seq[i+n_steps_in:i+n_steps_in+n_steps_out]
        X.append(x)
        Y.append(y)
        i+=1
    return np.array(X), np.array(Y)

def showRandRows(M,N,leg=False):
    '''
    funnction to sample rows from a matrix M, returns the N rows chosen if 'leg' 
    it will add legend to the plot
    '''s
    randRows  =np.random.randint(0,M.shape[0],N)
    for i in randRows:
        plt.plot(M[i])
    if leg:
        pl.legend(randRows)
    return randRows