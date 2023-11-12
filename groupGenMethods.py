import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import keras

COLORS = list(colors.CSS4_COLORS.keys())

def prep_data(filename = 'data/county1000.csv', exclude = [403,195]):
    '''
    custom data preprocessing for the 'county1000.csv' dataset. 
    This removes the data in the rows identified in the 'exclude'
    parameter. Weekends are removed, Fridays are stitched to Mondays.
    
    a different preprocessing function might be best for other sequential data.
    '''
    DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
    DAYS_SUM = [sum(DAYS[0:k]) for k in range(12)]
    #how many days have passed at the end of the ith month^^
    df = load_region(filename)
    data = np.asarray(df[DAYS_SUM[5]*24*4:DAYS_SUM[8]*24*4]).astype('float16').T
    # this grabs the june july aug data
    blds, t_len = data.shape
    num_days = t_len//(24*4)
    data_weeks = data[:,:91*24*4].reshape(blds,91//7,24*4*7 )
    weekdays = np.concatenate([data_weeks[:,:,:24*4],data_weeks[:,:,24*4*3:]],2).reshape(blds,-1)
    out = np.delete(weekdays, exclude, 0)
    # we remove bad rows/bad buildings
    return out

        

def k_clustering(data, k=13, runs=4):
    '''
    This will run the K-means clustering algorithm on the signals
    within the rows of 'data' with 'k' means 'runs' times to 
    obtain the lowest dbi score. Will return the cluster centers in the
    same dimensional space as the data, the labels matrix, and the dbi score
    for the optimal clustering
    
    This exists as the K-means clustering algorithm isn't deterministic
    and different runs thereof may create different clusters.
    '''
    bestdbi = 1e5
    bestclust = None
    for run in range(runs):
        clust = KMeans(k).fit(data)
        dbi = davies_bouldin_score(data, clust.labels_)
        if dbi<bestdbi:
            bestclust = clust
            bestdbi=dbi
    cntrs = bestclust.cluster_centers_
    lbls = bestclust.labels_
    print('dbi:',np.round(bestdbi,3))
    return (cntrs, lbls, bestdbi)


def exploreKs(data, low = 5, high = 20, runs = 4):
    '''
    This will run the 'k_clustering' function for
    with parameters 'k' and 'runs'. The 'k' parameter
    will be all integers from 'low' to 'high'. The returns
    of the best cluster from the 'k_clustering' function,
    judged by the davies-bouldin parameter, will be 
    returned.
    '''
    res = {}
    bestk=0
    bestdbi=1e5
    for k in range(low, high+1):
        print('k:', k, end=' ')
        res[k] = k_clustering(data,k=k, runs=runs)
        _, _, dbi = res[k]
        if dbi<bestdbi:
            bestdbi=dbi
            bestk = k
    print("\nWe'll go with "+str(bestk)+" clusters")
    return res[bestk]



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
        plt.legend(randRows)
    return randRows

def dispRandRows(M, N,figsize=(6,3),title=''):
    '''
    plots N2 rows in an array N2/4 by 4 of individual plots, where N2 is the smallest multiple
    of 4 equal to or greater than N.
    plots contain random rows of matrix M
    '''
    N2 = (N//4+1)*4
    randRows  =np.random.choice(M.shape[0],size=N2,replace=False)
    plt.subplots(N2//4,4,figsize=figsize)
    plt.suptitle(title)
    for i in range(N2):
        plt.subplot(N2//4,4,i+1)
        plt.plot(M[randRows[i]])
        plt.title(randRows[i])
    plt.tight_layout()


def tstr():
    '''
    provides a string of current time in PST
    for time sensitive file names with no colons or spaces
    '''
    out = ''
    for i in time.ctime():
        if i!=' ' and i!=':':
            out+=i
        else:
            out+='_'
    return out

def load_region(file): 
    '''
    read a csv file 
    and return a dataframe
    '''
    df = pd.read_csv(file)
    df.index = df['timestamp']
    del df['timestamp']
    return df

def params5(data):
    '''
    computes 5 parameters to represent each row of the data
    they are the 90th percentile, 10th percentile, high-load duration
    rise time and fall time.
    the high load duration is the time a signal spends above the average
    of the 90th and 10th percentile.
    the rise time is the time a signal takes to go from the
    10th percentile value to the 90th percentile value
    the fall time is the same, but going from the 90th percentile
    to the 10th percentile value.
    '''
    blds, t_len = data.shape
    num_days = t_len//(24*4)
    data_days = data.reshape(blds,num_days,24*4)
    two_days = np.zeros((blds,num_days-1,24*4*2))
    for i in range(num_days-1):
        two_days[:,i,:] = np.concatenate([data_days[:,i,:],data_days[:,i+1,:]],axis=1)
    n_peak = np.percentile(two_days,90,axis=2,keepdims=True)
    n_base = np.percentile(two_days,10,axis=2,keepdims=True)
    high_load_dur = np.sum((two_days>.5*(n_peak+n_base)).astype(int),axis=(1,2),keepdims=True)
    rise_times_days = np.zeros((blds,num_days-1))
    fall_times_days = np.zeros((blds,num_days-1))
    records = {}
    for b in range(blds):
        records[b] = []
        for d in range(num_days-1):
            arr = two_days[b,d,:]
            above_peak = (arr>=n_peak[b,d,0]).astype(int)
            below_base = (arr<=n_base[b,d,0]).astype(int) 
            above_ind = np.nonzero(above_peak)[0]
            below_ind = np.nonzero(below_base)[0]
            jumps,rise,fall = riseNfall(above_ind,below_ind)
            rise_times_days[b,d] = rise
            fall_times_days[b,d] = fall
            records[b].append(jumps)
    fall_times = np.mean(fall_times_days,axis=1)
    rise_times = np.mean(rise_times_days,axis=1)
    params = (
    np.concatenate([np.mean(n_peak,axis=1),
                   np.mean(n_base,axis=1),
                   np.mean(high_load_dur,axis=1),
                   (fall_times).reshape(blds,1),
                   (rise_times).reshape(blds,1)]
                  ,axis=1))
    p_means = np.mean(params,keepdims=True,axis=0)
    p_stdevs = np.std(params,keepdims=True,axis=0)
    return (params, p_means, p_stdevs)

def riseNfall(above,below):
    '''
    this is a helper function to the 'params5' function.
    it takes 'above' which holds the indices of a signal
    where it is above a threshold (90th percentile here)
    and 'below', the indices where the signal is below
    another threshold. 
    
    the returned values are a record of jumps
    bewteen the low and high threshold, the rise time
    and the fall time
    '''
    above,  below , jumps = list((above)), list((below)), []
    ai,bi = 0,0
    start = True
    while (ai < len(above) and bi < len(below)):
        if above[ai] < below[bi]:
            jumps += [[above[ai], 1]]
            ai += 1
        else:
            jumps += [[below[bi], -1]]
            bi += 1
    jumps = np.asarray(jumps)
    rise,fall = 0,0
    for j in range(1,len(jumps)):
        if jumps[j,1] + jumps[j - 1, 1] == 0:
            diff = jumps[j,0] - jumps[j - 1, 0]
            if jumps[j, 1] < 0:
                fall = (diff)
            else:
                rise = (diff)
        if rise > 0 and fall > 0:
            break
    return jumps, rise,fall

def norm(arr, indx=1,uniform=False, a=-1,b=1):
    ''' 
    function to normalize data along 'index' axis by centering  at 0, and
    dividing by standard deviation.
    'uniform' will rescale the data such that the maximum is b and minimum is a
    the relevant parameters of the original data are also returned with the normalize array
    (normalized_array, mean, st deviation) or (normalized_array, min, max)
    '''
    if not uniform:
        s = np.std(arr,indx,keepdims=True)
        mu = np.mean(arr,indx,keepdims=True)
        out = (arr-mu)/s
        return (out,mu,s)
    else:
        mx = np.max(arr,indx,keepdims=True)
        mn = np.min(arr,indx,keepdims=True)
        out = 2*(arr-mn)/(mx-mn) - 1
        return (out,mn,mx)

    
def outClusterBlds(cluster=None,bldCenters=None,data=None,labels=None):
    '''
    returns an array collecting signals
    from data coming from every cluster except cluste 'cluster'.
    Each cluster contributes the data closest to its cluster center.
    This is meant to give wrong examples of what we want to a GAN.
    'cluster' is the index of the cluster that we want to imitate.
    'bldCenters' is a list of row indices such that the ith value
    is the row index in 'data' whose signal's clustering parameters
    are closest to the center of cluster i.
    'data' is the relevant dataset, separated by rows of signals. 
    This is the dataset that was clustered.
    'labels'
    '''
    out = []
    for i in range(len(set(labels))):
        if i != cluster:
            out.append(data[bldCenters[i]])
    return np.asarray(out)      

def clusterCenterBlds(centers=None, prms = None):
    '''
    returns a list of row indices where the ith value
    in the list is the row in 'prms' which is closest to
    the ith center in 'centers'. This means that in the relevant
    dataset, the same row will be that which most closely
    represents the data in cluster i.
    'centers' holds the vectors defining each cluster's center from the
    K-means clustering algorithm. 'prms' holds the clustering
    parameters for the data that is clustered. Each row in 'prms'
    holds the clustering parameters for the same row
    in the original data.
    '''
    dists = (np.sum(np.square(prms),1,keepdims=True) 
             - 2 * prms.dot(centers.T) 
            + np.sum(np.square(centers),1,keepdims=True).T)
    repBlds = np.argmin(dists,0)
    return repBlds


def compParams(gen, prms, cluster, shape = (8,8)):
    '''
    plotting function to compare the parameters of the original
    data, 'prms', and those of the generated data, 'gen',
    in the space used for clustering. The first plot will show
    squared distances to cluster centers per dimension. 
    The second plot shows L2 distance to each cluster.
    '''
    print("we'd like to be close to cluster", cluster)
    sqDist = np.square(gen- prms)
    l2 = np.sqrt(np.sum(sqDist,axis=1))
    plt.figure(figsize=shape)
    dims = prms.shape[1]
    for dim in range(dims):
        plt.plot(sqDist[:,dim])
    plt.xlabel('Clusters')
    plt.ylabel('Squared Distance')
    plt.axvline(cluster, dashes=(2,4))
    plt.title('Distance to Clusters per Clustering Dimension')
    plt.legend(['dim '+str(i) for i in range(1,dims+1)])
    plt.show()
    plt.figure(figsize=shape)
    plt.plot(l2)
    plt.xlabel('Clusters')
    plt.ylabel('L2 Distance')
    plt.axvline(cluster, dashes=(2,4))
    plt.title('L2 Distance to Clusters')
    plt.show()
    return sqDist, l2

def makeWrongDays(cluster=None, bldCenters=None, data=None, labels=None):
    ''' 
    return a long list of daily signals representing the other clusters for training a
    GAN to make signal resembling a specific cluster.
    'cluster' is the number of the cluster to try to imitate later.
    'bldCenters'
    '''
    out = outClusterBlds(cluster=cluster,bldCenters=bldCenters,data=data,labels=labels)
    wronglist = []
    for bldarr in out:
        #we essentially grab each building representing other clusters
        #and taking that normalized signal and splitting it into 96 dimensional vectors
        #then, these are all combined to have a big array of days that are legit
        #power signals, but from other clusters
        x,y = splitSeqs(bldarr,0,24*4)
        wronglist.append(y)
    wrongdays = np.asarray(wronglist).reshape(-1,96)
    return wrongdays

def generate_wrong_samples(wrongdays=None, n=50):
    '''
    given the 'wrongdays' array, grab 'n' columns from it
    to be used as counterexamples for what we want from a GAN.
    '''
    chosen =  np.random.choice(wrongdays.shape[0],n)
    # pick 'n' random indices less than the number of rows
    # create class labels
    X = wrongdays[chosen]
    rows, cols= X.shape
    X = X.reshape(rows,cols,1)
    y = np.zeros((n, 1))
    return X, y        
    
    
# train a generative adversarial network on a one-dimensional function
def define_discriminator(n_inputs=24*4,lr = 1e-3, arch = 'c4 c4 f d48 d18 d9'):
    '''
    returns a compiled tensorflow model with custom dense and 1D convolutional 
    layers controlled by the 'arch' parameter. ReLu activation functions and outputting
    a single value from a sigmoid function. Designed to be used as a discriminator for
    a GAN. 
    'lr' is the learning rate. 
    'n_inputs' is the length of the input array
    'arch' encodes the architecture. 
    a 'c4' denotes a convolutional layer with filter size 4
    an 'f' denotes a flatten layer
    a 'd10' denotes a dense layer with 10 cells.
    '''
    count = 1
    model = Sequential()
    for layer in arch.split():
        if layer=='f':
            model.add(Flatten(name='f'+str(count))) 
        elif 'c' in layer:
            kernel_size = int(layer[1:])
            model.add(Conv1D(1, kernel_size, activation='relu', name = "c"+str(count)))
        elif 'd' in layer:
            cells = int(layer[1:])
            model.add(Dense(cells, activation='relu', kernel_initializer='he_uniform', 
                            name = "d"+str(count)))
        count+=1
    model.add(Dense(1, activation='sigmoid', name = "d_final"))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  metrics=['accuracy'])
#     model.summary()
    return model
 
# define the standalone generator model
def define_generator(latent_dim, n_outputs=24*4, in_shape=None, arch = '16 16 24'):
    '''
    returns a custom tensorflow model that takes an array of length 'latent_dim' and
    generates another array of length 'n_outputs'. This is meant to be used in a 
    GAN. The 'in_shape' parameter simply includes more details about the array to
    be taken as input, as tensorflow near mandates that it be a 3 axis tensor. 
    See usage in demo code for more details on 'in_shape'.
    'arch' is the blueprint for the generator. It's a string of numbers
    separated by spaces and each number will become a dense layer
    with number of cells equal to its value. The default '16 16 24' thus makes 
    a model three dense layers with 16, 16 and 24 cells in that order. 
    
    the model's output is generated with a convolutional 1D transpose layer
    stacked on top of the last dense layer.
    '''
    model = Sequential()
    count=1
    for cells in arch.split():
        model.add(Dense(int(cells), activation='tanh', kernel_initializer='he_uniform',
                    input_shape=in_shape, name = "fc"+str(count)))
        count+=1
    model.add(Conv1DTranspose(1, 96-latent_dim+1, name = "ct_final"))
    return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator, lr=1e-3):
    '''
    makes a GAN from the 'generator' and 'discriminator' models
    provided. 'lr' is the learning rate used.
    '''
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
#     model.summary()
    return model
 
# generate n real samples with class labels
def generate_real_samples(x,n):
    '''
    grabs 'n' random rows from the dataset 'x'
    uses them to generate X,y training
    pairs for a discriminator model
    '''
    # generate inputs in [-0.5, 0.5]
    np.random.shuffle(x)
    X = x[:n,:,:]
    # generate class labels
    y = np.ones((n, 1))
    return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    '''
    generates random values to be used as inputs to generator
    '''
    #made a method to allow future editing
    return np.random.randn(n,latent_dim,1)
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    '''
    utilizes the 'generator' model to make 'n' fake signals.
    These will be used as training data for the discriminator in the GAN.
    'latent_dim' is the input size to the generator.
    '''
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = np.zeros((n, 1))
    return X, y
 
# evaluate the discriminator and plot real and fake curves
def summarize_performance(epoch, generator, discriminator, latent_dim, n=20,
                          M=None, saveplot=False,label='', savecurve=False):
    '''
    a progress checking function that plots samples of true signals from the 
    desired cluster. The method then plots signals generated by the GAN for visual 
    comparison.
    'epoch' is the epoch in training where we are
    'generator' and 'discriminator' refer to the GAN's models
    'latent_dim' refers to the input length for vectors inputted to the
    generator
    'n' is the number of curves to generate from the GAN, see 'savecurve' parameter
    'M' is the dataset.
    'saveplot' is a boolean determining whether or not to save images of the plots made
    'label' is a string to uniquely labe the files created
    'savecurve' is a boolean determining whether to save the generated curves
    
    '''
    # prepare real samples
    x_real, y_real = generate_real_samples(M,n)
    
    # evaluate discriminator on real examples
    loss_real, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    if savecurve:
        np.save('records/'+label+'.npy',x_fake)
    # evaluate discriminator on fake examples
    loss_fake, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('EPOCH: ',epoch,'  REAL ACC: ', acc_real,'  FAKE ACC: ', acc_fake)
    print('xreal ',x_real.shape, 'x_fake' ,x_fake.shape)

    dispRandRows(x_real,12, figsize=(8,4), 
                 title='real signals at '+str(epoch)+' epochs')
    plt.show()
    dispRandRows(x_fake,12, figsize=(8,4),
                 title='fake signals at '+str(epoch)+' epochs')
    if saveplot:
        plt.savefig('records/'+label)
    plt.show()
    return

# evaluate the discriminator and plot real and fake curves
def save_progress(epoch, generator, discriminator, latent_dim,
                  x_real, y_real,x_fake, y_fake, x_wrong, y_wrong,
                  progress = [], label=''
                 ):
    '''
    a record keeping function that will generate a csv file 
    with the losses and accuracies of the discriminator on
    real data, fake data, and wrong data represented as r, f, and w, 
    respectively. 
    the 'epoch' in training is also saved. a 'label' can be included to make
    the name of the csv file unique. the list 'progress' can have previous 
    values of the loss and accuracies previously described so that the csv 
    can track performance across layers.
    'x_real', 'y_real', 'x_fake', 'y_fake', 'x_wrong', and 'y_wrong'
    are input, output pairs for supervised learning consisting of
    real, fake and wrong data for the GAN
    '''
    # evaluate discriminator on real examples
    loss_real, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # evaluate discriminator on fake examples
    loss_fake, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    loss_wrong, acc_wrong = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    progress.append([epoch,loss_real, acc_real,loss_fake, acc_fake, loss_wrong, acc_wrong])
    df = pd.DataFrame(progress,columns=['epoch','r_loss','r_acc','f_loss','f_acc','w_loss', 'w_acc'])
    df.to_csv('records/'+label+'.csv')
    return
# train the generator and discriminator

def train(g_model, d_model, gan_model, latent_dim, n_epochs=1000,
          n_batch=128, n_eval=100, M=None, label='', wrong_signals=None):
    '''
    this is a wrapper function to handle the training of a generator 'g_model',
    a discriminator 'd_model' and their respective GAN 'gan_model'. 
    'latent_dim' is the length of vector that is inputted to the GAN.
    'n_epochs' is the number of epochs to train the GAN
    'n_batch' is the batch size to use for training the models
    'n_eval' is the number of epochs after which to have the
    GAN do a progress check. See the 'summarize_performance' function
    'M' is the array containing our dataset. Relevant signals are in the rows.
    'label' is a string so that progress checks and total results
    can be saved with a unique filename.
    'wrong_signals' is a collection of real data coming from a cluster 
    other than the one we're trying to imitate. These will be used as counterexamples 
    to the GAN
    '''
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    progress=[]
    # manually enumerate epochs
    t0 = time.time()
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(M,n_batch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_batch//2)
        x_wrong, y_wrong = generate_wrong_samples(wrong_signals,n_batch)
        x_all , y_all = np.concatenate([x_real, x_fake, x_wrong]), np.concatenate([y_real, y_fake, y_wrong])
        # double the data and epochs for discriminator as
        # its performance is the lower bound for our GAN's success
        # update discriminator
        d_model.fit(x_all, y_all, verbose=0, epochs =2)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.fit(x_gan, y_gan, verbose=0)
        # evaluate the model every n_eval epochs
        if (i % n_eval == 0 or i+1==n_epochs or i== 5) and (i!=0):
            if i+1!=n_epochs:
                summarize_performance(i, g_model, d_model, latent_dim,M=M)
            else:
                summarize_performance(i, g_model, d_model, latent_dim,M=M,
                                      saveplot=True, label=label, savecurve=True)
            dt = time.time()-t0
            print((label+'\n')*2,"We're doing about 200 epochs in ", 
                  np.round(dt*200/((i+1)*60),2), ' min')
        if i+1==n_epochs:
            save_progress(i, g_model, d_model, latent_dim,
                  x_real, y_real,x_fake, y_fake,x_wrong,y_wrong ,progress = progress, label=label) 
    timestr = tstr()[4:-8]

    return

