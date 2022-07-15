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

def showRandRows(M,N=None,leg=False):
    '''
    function to sample rows from a matrix M, returns the N rows chosen if 'leg' 
    it will add legend to the plot
    '''
    if N==None:
        N = M.shape[0]
    randRows  =np.random.choice(M.shape[0],size=N,replace=False)
    for i in randRows:
        plt.plot(M[i])
    if leg:
        pl.legend(randRows)
    return randRows

def dispRandRows(M, N,figsize=(6,3)):
    '''
    plots N2 rows in an array N2/4 by 4 of individual plots, where N2 is the smallest multiple
    of 4 equal to or greater than N.
    plots contain random rows of matrix M
    '''
    N2 = (N//4+1)*4
    randRows  =np.random.choice(M.shape[0],size=N2,replace=False)
    plt.subplots(N2//4,4,figsize=figsize)
    for i in range(N2):
        plt.subplot(N2//4,4,i+1)
        plt.plot(M[i])
    plt.tight_layout()
