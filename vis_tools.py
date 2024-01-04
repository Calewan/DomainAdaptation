# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:11:54 2023

@author: savci
"""
"""
After training Networks in 'training_loop.py', evaluate their performance
Main functions: 
            visualizer       (Calculate and Visualize all kinds of metrics;
                              For calling and details simply use 
                              'help visualizer')
            
            vis_tsne_and_kde (Additionally to dimensionality reduction in 
                              'visualizer' create 2 Dimensional tsne plots 
                              with more information about the class distribution 
                              using a gaußian kernel density estimate additionally
                              to the simple scatter plot)
            """




import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import sklearn.neighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from scipy import stats
from scipy.sparse.csgraph import shortest_path
#from scipy.stats import gaussian_kde
from initialize_network import *
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
#import tensorflow.keras.backend as K
from matplotlib.colors import LinearSegmentedColormap
from data_creators import data_create

'''
    Calculations are fitting DANN but no other Networks right now. 
    The dimensions and layer adresses are hardcoded
'''
def dann_manifold_disentanglement(classifier, X, y, k, label,depth):    
    
    '''
    computes the weighted distance between geodesic distance and euclidean distance of all points of the given label
    
    classifier: keras model
    X: input features
    y: classes corresponding to X, NOT ONE HOT
    k: hyperparamter of the kneighbors_graph algorithm, if it is to small, NaN values will appear since
    no path between points is found; if it is too large, the approximation of the geodesic distance will be bad
    
    label: class for which the distance should be computed
    OUTPUT:
    d   : class entanglement
    siz : percentage of values considered (due to points sorted out if they are too close to each other) 
    '''
    
    def calculate_d(euk_dist_tri, geo_dist_tri):
        '''
        calculates the weighted distance from the distances of all considered points
        '''
        N = euk_dist_tri.shape[0]
        summ = 0
        for i in range(N-1): # zeilen
            summ += np.sum(np.divide(np.abs(euk_dist_tri[i,i+1:]-geo_dist_tri[i,i+1:]),np.abs(euk_dist_tri[i,i+1:])))
        d =  2/(N*(N-1)) * summ
        return(d)
    def calculate_d_alpha_xi(euk_dist_tri, geo_dist_tri, alpha, xi):
        '''
        calculates the weighted distance from the distances of all considered points
        '''
        N       = euk_dist_tri.shape[0]
        upper   = np.triu_indices(N)
        eu      = euk_dist_tri[upper].reshape(-1,)#pairwise euklidean
        ri      = geo_dist_tri[upper].reshape(-1,)#pairwise rieman/geodesic 
        ri      = ri[eu > xi]
        eu      = eu[eu > xi]
        siz     = len(eu) *2/(N*(N-1))
        vals    = np.divide(np.abs(eu-ri),eu)
        #len_N = np.floor(np.shape(xi_trimmed)[0]*(1-alpha)).astype('integer')
        #alpha_trimmed = np.divide(np.sum(xi_trimmed[:len_N]),len_N)
        d       = stats.trim_mean(vals, alpha/2)
        return d, siz
    
    h_current   = X[y == label,:,:,:] # the feature vectors of the corresponding label
    layer_idx   = 0
    # iterates through layers to compute the distance for each layer
    idid        = 0
    d           = [0]* (10)
    siz         = [0]* (10)
    print("_____________________________________________")
    print("_____________label %f__________________"%label)
    #neighbors_graph = sklearn.neighbors.kneighbors_graph(X = h_current, n_neighbors = k, mode='distance')
    # computes the shortest distance between each pair of points w.r.t. the neighbors_graph
    #geo_dist_matrix = shortest_path(neighbors_graph, method='auto', directed=False)
    ##euklidean distance
    #euk_dist_matrix = euclidean_distances(h_current)
    #mypca(h_current, Y)
    #d[idid]     = calculate_d(euk_dist_matrix, geo_dist_matrix)
    h_current = classifier.feature_extractor_layer0(h_current)
    h_current = classifier.feature_extractor_layer1(h_current)
    h_current = classifier.feature_extractor_layer2(h_current)
        
    h_current = classifier.feature_extractor_layer3(h_current)
    h_current = classifier.feature_extractor_layer4(h_current)
    h_current = classifier.feature_extractor_layer5(h_current)
    h_current = classifier.feature_extractor_layer6(h_current)
        
    feature = tf.reshape(h_current, [-1, 4 * 4 * 64]) # apply the layer to the current data
    lp_x = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])
    ids  = [0,1,3,5,7,9,11,13,15,17,19]#usw
    for i in range(depth):
        neighbors_graph     = sklearn.neighbors.kneighbors_graph(X = lp_x, n_neighbors = k, mode='distance')
        # computes the shortest distance between each pair of points w.r.t. the neighbors_graph
        geo_dist_matrix     = shortest_path(neighbors_graph, method='auto', directed=False)
        euk_dist_matrix     = euclidean_distances(lp_x)
        #trimmed mean(xi) or full data(no xi) set mean:
       # d[idid]     = calculate_d(euk_dist_matrix, geo_dist_matrix)
        d[idid],siz[idid]    = calculate_d_alpha_xi(euk_dist_matrix, geo_dist_matrix,alpha = 0.1, xi=1)
        print("class. layer {}: distance of euclidean and geodesic distance is {}".format(layer_idx, d[idid]))
        layer_idx   += 1
        idid        += 1
        lp_x         = classifier.layers[7+ids[i]](lp_x)
        # computes a graph connecting each point to its k neighbors to approximate the geodesic
    neighbors_graph     = sklearn.neighbors.kneighbors_graph(X = lp_x, n_neighbors = k, mode='distance')
            # computes the shortest distance between each pair of points w.r.t. the neighbors_graph
    geo_dist_matrix     = shortest_path(neighbors_graph, method='auto', directed=False)
    euk_dist_matrix     = euclidean_distances(lp_x)
   # d[idid]     = calculate_d(euk_dist_matrix, geo_dist_matrix)
    d[idid],siz[idid]   = calculate_d_alpha_xi(euk_dist_matrix, geo_dist_matrix,alpha = 0.1, xi=1) # xi = min distance between points, alpha = trimm extreme values
    
    print("class. layer {}: distance of euclidean and geodesic distance is {}".format(layer_idx, d[idid]))
    print("---------------------------------------------------------")
    return d, siz

def manifold_disentanglements_unprocessed_data( X, y, k, label,depth):
     def calculate_d_alpha_xi(euk_dist_tri, geo_dist_tri, alpha=0.1, xi=1):
         '''
         calculates the weighted distance from the distances of all considered points
         '''
         N = euk_dist_tri.shape[0]
         upper = np.triu_indices(N)
         eu = euk_dist_tri[upper].reshape(-1,)#pairwise euklidean
         ri = geo_dist_tri[upper].reshape(-1,)#pairwise rieman/geodesic 
         ri = ri[eu> xi]
         eu = eu[eu>xi]
         siz = len(eu) *2/(N*(N-1))
         vals = np.divide(np.abs(eu-ri),eu)
         d = stats.trim_mean(vals, 0.05)
         return d, siz
     h_current = X[y == label,:] # the feature vectors of the corresponding label
     print("_____________________________________________")
     neighbors_graph = sklearn.neighbors.kneighbors_graph(X = h_current, n_neighbors = k, mode='distance')
     geo_dist_matrix = shortest_path(neighbors_graph, method='auto', directed=False)
         ##euklidean distance
     euk_dist_matrix = euclidean_distances(h_current)

     d,c     = calculate_d_alpha_xi(euk_dist_matrix, geo_dist_matrix)
     N = euk_dist_matrix.shape[0]
     print("layer {}, label{}: distance of euclidean and geodesic distance is {}".format(depth,label, d))
     print("of {} combinations, some were discarded because the eucl dist was too small, {}% were used for calculation".format(N*(N-1)/2,c))
     return d, c
def classify_domain(inputsize):
    '''Intitialize simple domain classification network'''
    model = Sequential()
    model.add(Dense(32,activation='relu',input_shape=(inputsize,)))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.5))      
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def load_model_from_data(source_domain=0,target_domain=0,depth=4,snr_s=3,channel=1,DA = True):
    '''Load trained models and Data'''
    reconstructed_model = DANN(depth = depth+1)
    #reconstructed_model.build(input_shape = (1,28,28,3))
    images = np.zeros((5,28,28,channel))
    if DA == True:
        reconstructed_model(images, train=False, source_train=False)
    else:
        reconstructed_model(images, train=False, source_train=True)


    if DA == True:
        if source_domain == target_domain:
            if source_domain == 0:
                reconstructed_model.load_weights('trained_networks_DomAdapt\DANN_S' + str(source_domain) +'_T' + str(1) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
            else:
                reconstructed_model.load_weights('trained_networks_DomAdapt\DANN_S' + str(source_domain) +'_T' + str(0) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
                
            #get test data
            _,_, mnist_m_test_x, mnist_m_test_y = data_create(source_domain, source_domain, snr = snr_s, plotsamples = False)
        else:
            reconstructed_model.load_weights('trained_networks_DomAdapt\DANN_S' + str(source_domain) +'_T' + str(target_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
            #reconstructed_model.load_weights('/p/sys/department/Masterarbeit_Can/MTHESIS/AlexPaperCode/trained_networks/DANN_S' + str(source_domain) +'_T' + str(target_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'da.h5')
            #get test data
            _,_, mnist_m_test_x, mnist_m_test_y = data_create(target_domain, target_domain, snr = snr_s, plotsamples = False)
            
    else:
        reconstructed_model.load_weights('trained_networks_SourceOnly\DANN_S' + str(source_domain) + '_d'+ str(depth)+ '_snr'+str(snr_s)+'source.h5')
        _,_, mnist_m_test_x, mnist_m_test_y = data_create(target_domain, target_domain, snr = snr_s, plotsamples = False)

    #calculate accuracy :)
    if mnist_m_test_x.shape[3] < channel:
        mnist_m_test_x = np.repeat(mnist_m_test_x, channel, axis=3)
    if DA == True:
        output = reconstructed_model(mnist_m_test_x, train=False, source_train=True)
    else:
        output = reconstructed_model(mnist_m_test_x, train=False, source_train=True)  
    
    y_pred = output.numpy()
    y_pred = np.argmax(y_pred, axis = 1)
    acc  = accuracy_score(mnist_m_test_y, y_pred)
    return reconstructed_model,acc

def selfmade_backend(x,model,ind,DA):
    '''Calculate intermediate data representations'''
    x = x.reshape(x.shape[0],28,28,-1)
    for i in range(ind+1):
        x = model.layers[i](x)
        if i == 6:
            x = tf.reshape(x, [-1, 4 * 4 * 64])
            #x = x
    return x

def vis_tsne_and_kde(depth = 4, DA = False):
    '''
        Plot a 2D t-SNE Visualization of the Data/Representations as scatter and 
        Gaussian KDE thereof for all Combinations of Source and Target Domain
        DA : True  : Consider Damain Adaptation Training, i.e. DANN; 
             False : Consider Source-Only Training
  
        TODO: Make the code readable...
    '''
    
    #depth = 4
    I_source = [0,1,2,3,4,5]
    I_target = [0,1,2,3,4,5]
    accs = np.zeros((len(I_source),len(I_target)))
    B = [6,7,8,10,12,14,16]
    b = B[depth]
    
    accs_discr = np.zeros((len(I_source),len(I_target)))
    d = np.zeros((6,6,10,10))
    d_t = np.zeros((6,6,10,10))
    per = np.ones((6,6,10,10))
    per_t = np.ones((6,6,10,10))
    snr_s = 3
    C = [['xkcd:forest green','xkcd:blue','xkcd:turquoise','xkcd:violet','xkcd:green','xkcd:mustard','xkcd:blue green','xkcd:orange','xkcd:hot pink','xkcd:cherry red'],['xkcd:olive','xkcd:light blue','xkcd:aqua','xkcd:light purple','xkcd:pale green','xkcd:beige','xkcd:grey blue','xkcd:light orange','xkcd:light pink','xkcd:pinkish red']]
    data_color_s = sns.xkcd_palette(['forest green','blue','turquoise','violet','green','mustard','blue green','orange','hot pink','cherry red'])#s = 0
    data_color_t = sns.xkcd_palette(['olive','light blue','aqua','light purple','pale green','beige','grey blue','light orange','light pink','pinkish red'])
 
    for s in I_source:
        for t in I_target:

            if s== t:#No DA
                mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = data_create(s, s, snr = snr_s, plotsamples = False)
                Cha = [1,1,3,3,3,1]
                if DA == True:
                    channel = Cha[s]
                else:
                    channel = 3
                    if mnist_train_x.shape[3] < channel:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                mnist_train_x   = mnist_train_x.astype('float32')
                mnist_test_x    = mnist_test_x.astype('float32')
                mnist_train_y_dec = mnist_train_y
                mnist_test_y_dec  = mnist_test_y
                mnist_train_y = to_categorical(mnist_train_y)
                mnist_test_y = to_categorical(mnist_test_y)
                classifier, accs[s,t]=load_model_from_data(s,t,depth,snr_s,channel,DA = DA)
                classifier.build(input_shape = (1,28,28,channel))
                #classifier.summary()
                print(classifier.layers[b].name)
                if plot_entangle == True:
                    for i in range(10):
                        d[s,t,i,:],per[s,t,i,:] = dann_manifold_disentanglement(classifier, mnist_train_x  , mnist_train_y_dec, 15, i,depth)
                if dim_red == True:
                    discr_train_x = mnist_train_x[:10000,:,:,:]
                    YYY     = mnist_train_y_dec[:10000]
                    pca_50 = PCA(n_components=50)
                    discr_train_x = discr_train_x.reshape(-1,28*28*channel)
                    pca_result_50 = pca_50.fit_transform(discr_train_x)
                    x   = pca_result_50[:, 0]
                    x1  = pca_result_50[:10000, 0]
                    y1 = pca_result_50[:10000, 1]
                    if pca_plot == True:
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        f, axs = plt.subplots(1,2, figsize=(8, 4))
                        plt.suptitle('PCA Unprocessed Data - var='+str(var)+'%, Domain '+str(s))
                        #for ii in range(10):
                        sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                            levels=5, ax=axs[0], legend=False)
                        axs[0].axis('off')
                        for ii in range(10):
                            scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, alpha = 0.75)#,edgecolors = "grey", linewidths = 0.1 )
                        plt.axis('off')
                        plt.savefig('Example\dim_red\s_sns_kde\SOPCAUnprocessedDatavar'+str(var)+'Domain'+str(s)+'.pdf')
                        plt.show()

                    if tSNE_plot == True:
                        tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                        X_transformed = tsne.fit_transform(pca_result_50)
                        # Plot t-SNE outputs
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        f, axs = plt.subplots(1,2, figsize=(8, 4))
                        sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                            levels=5, ax=axs[0], legend=False)
                        axs[0].axis('off')
                        for ii in range(10):
                            scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, s = 5, alpha = 0.75)
                        plt.suptitle('t-SNE Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        #plt.savefig('Example\dim_red\da_sns_kde\'
                        plt.savefig('Example\skde\SOtSNEUnprocessedDataDomain'+str(s)+'.pdf')
                        plt.show()
                    if isomap_plot == True:
                        isomap = Isomap(n_components=2)
                        X_transformed = isomap.fit_transform(pca_result_50)
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#,s = 3,alpha = 0.75)
                        plt.title('Isomap Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        plt.show()    
                    if LLE_plot == True:
                        lle = LocallyLinearEmbedding(n_components=2)
                        X_transformed = lle.fit_transform(pca_result_50)
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                        plt.title('LLE Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        plt.show()    
                        
                    for i in range(depth):
                        #funce = selfmade_backend(x,classifier,B[i])#K.function([classifier.layers[0].input], [classifier.layers[B[i]].output])
                        representation = selfmade_backend(discr_train_x,classifier,B[i],DA)#funce(discr_train_x)
                        representation = representation.numpy()
                        rep_plot       = representation[:10000,:]#Random
                        pca_50 = PCA(n_components=50)
                        pca_result_50 = pca_50.fit_transform(rep_plot)
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        
                        if pca_plot == True:
                            x1 = pca_result_50[:10000, 0]
                            y1 = pca_result_50[:10000, 1]
                            f, axs = plt.subplots(1,2, figsize=(8, 4))
                            sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                                levels=5, ax=axs[0], legend=False)
                            axs[0].axis('off')
                            for ii in range(10):
                                scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, s = 3, alpha = 0.75)
                            plt.suptitle('PCA Dense Layer '+str(i)+' - var='+str(var)+'%, Domain '+str(s))
                            plt.axis('off')
                            plt.savefig('Example\dim_red\s_sns_kde\SOPCADenseLayer'+str(i)+'var'+str(var)+'%Domain'+str(s)+'.pdf')
                            plt.show()
                        if tSNE_plot == True:
                            tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                            X_transformed = tsne.fit_transform(pca_result_50)
                            # Plot t-SNE outputs
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            f, axs = plt.subplots(1,2, figsize=(8, 4))
                            sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                                levels=5, ax=axs[0], legend=False)
                            axs[0].axis('off')
                            for ii in range(10):
                                scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, s = 3, alpha = 0.75)
                            plt.suptitle('t-SNE Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.savefig('Example\skde\SOtSNEDenseLayer'+str(i)+'Domain'+str(s)+'.pdf')
                            plt.show()
                        if isomap_plot == True:
                            isomap = Isomap(n_components=2)
                            X_transformed = isomap.fit_transform(pca_result_50)
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('Isomap Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.show()    
                        if LLE_plot == True:
                            lle = LocallyLinearEmbedding(n_components=2)
                            X_transformed = lle.fit_transform(pca_result_50)
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('LLE  Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.show()        
            else:
                #DA=True
                mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = data_create(s, s, snr = snr_s, plotsamples = False)
                mnist_m_train_x, mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(t, t, snr = snr_s, plotsamples = False)
                Cha = [1,1,3,3,3,1]
                if DA == True:
                    print('ha')
                    channel = max(Cha[s],Cha[t])
                    if Cha[s] < Cha[t]:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                    elif Cha[s] > Cha[t]:
                        mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                        mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
                else:
                    channel = 3
                    if mnist_train_x.shape[3] < channel:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                    if mnist_m_train_x.shape[3] < channel:
                        mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                        mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
                #inputsize = 100    #mnist_test_x[1]*mnist_test_x[2]*mnist_test_x[3]
                mnist_train_x   = mnist_train_x.astype('float32')
                mnist_test_x    = mnist_test_x.astype('float32')
                mnist_m_train_x = mnist_m_train_x.astype('float32')
                mnist_m_test_x  = mnist_m_test_x.astype('float32')
                mnist_train_y_dec = mnist_train_y
                mnist_test_y_dec  = mnist_test_y
                mnist_train_y = to_categorical(mnist_train_y)
                mnist_test_y = to_categorical(mnist_test_y)
                mnist_m_train_y_dec = mnist_m_train_y
                mnist_m_test_y_dec  = mnist_m_test_y
                mnist_m_train_y = to_categorical(mnist_m_train_y)
                mnist_m_test_y = to_categorical(mnist_m_test_y)

                #load model
                classifier, accs[s,t]=load_model_from_data(s,t,depth,snr_s,channel = channel,DA = DA)
                classifier.build(input_shape = (1,28,28,channel))
                #classifier.summary()
                print(classifier.layers[b].name)
                #func = selfmade_backend(x,classifier,b)
                
                #pca, t-sne, isomap, lle 
                if dim_red == True:
                    discr_train_x   = np.concatenate([mnist_train_x[:20000,:,:,:],mnist_m_train_x[30000:50000,:,:,:]])
                    #discr_train_lab = np.concatenate([mnist_train_y[:20000,:],mnist_m_train_y[30000:50000,:]])
                    YYY             = mnist_train_y_dec[:20000]
                    YYYY            = mnist_m_train_y_dec[30000:50000]
                    pca_50          = PCA(n_components=50)
                    discr_train_x   = discr_train_x.reshape(-1,28*28*channel)
                    pca_result_50   = pca_50.fit_transform(discr_train_x)
                    x               = pca_result_50[:, 0]
                    x1              = pca_result_50[:20000, 0]
                    x2              = pca_result_50[20000:, 0]
                    y1              = pca_result_50[:20000, 1]
                    y2              = pca_result_50[20000:, 1]
                    tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                    X_transformed = tsne.fit_transform(pca_result_50)
                    # Plot t-SNE outputs
                    x1 = X_transformed[:20000, 0]
                    x2 = X_transformed[20000:, 0]
                    y1 = X_transformed[:20000, 1]
                    y2 = X_transformed[20000:, 1]
                    f, axs = plt.subplots(1,2, figsize=(8, 4))
                    sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                        levels=5, ax=axs[0], legend=False)
                    sns.kdeplot(x=x2, y=y2, hue=YYYY,palette = data_color_t,
                        levels=5, ax=axs[0], legend=False)
                    axs[0].axis('off')
                    for ii in range(10):
                        scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, s = 5, alpha = 0.75)
                        scatter = axs[1].scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii], s = 5)#,s = 5,alpha = 0.75)#,edgecolors='gray')
                    plt.suptitle('t-SNE Unprocessed Data, Source '+str(s)+', Target '+str(t))
                    plt.axis('off')
                    plt.savefig('Example\skde\SOtSNEUnprocessedDataSource'+str(s)+'Target'+str(t)+'.pdf')
                    plt.show()                       
                    for i in range(depth):
                        #funce = selfmade_backend(x,classifier,B[i])#K.function([classifier.layers[0].input], [classifier.layers[B[i]].output])
                        representation = selfmade_backend(discr_train_x,classifier,B[i],DA)#funce(discr_train_x)
                        representation = representation.numpy()
                        rep_plot       = representation[:40000,:]#Random
                        pca_50 = PCA(n_components=50)
                        pca_result_50 = pca_50.fit_transform(rep_plot)
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                        X_transformed = tsne.fit_transform(pca_result_50)
                        # Plot t-SNE outputs
                        x1 = X_transformed[:20000, 0]
                        x2 = X_transformed[20000:, 0]
                        y1 = X_transformed[:20000, 1]
                        y2 = X_transformed[20000:, 1]
                        f, axs = plt.subplots(1,2, figsize=(8, 4))
                        sns.kdeplot(x=x1, y=y1, hue=YYY,palette = data_color_s,
                            levels=5, ax=axs[0], legend=False)
                        sns.kdeplot(x=x2, y=y2, hue=YYYY,palette = data_color_t,
                            levels=5, ax=axs[0], legend=False)
                        axs[0].axis('off')
                        for ii in range(10):
                            scatter = axs[1].scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii], s = 5)#, s = 5, alpha = 0.75)
                            scatter = axs[1].scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii], s = 5)#,s = 5,alpha = 0.75)#,edgecolors='gray')
                        plt.suptitle('t-SNE Dense Layer '+str(i)+', Source '+str(s)+', Target '+str(t))
                        #ax.legend()
                        plt.axis('off')
                        plt.savefig('Example\skde\SOtSNEDenseLayer'+str(i)+'Source'+str(s)+'Target'+str(t)+'.pdf')  
                        plt.show()
    return None

def visualizer(depth = 4, DA = False,show_accs = True,plot_domain_disc = True, plot_entangle = False,dim_red = False, tSNE_plot = False, pca_plot = False, isomap_plot=False, LLE_plot = False):
    ''' Visualize Networks and Data for all Combinations of Source and Target Domain
        DA                  : True  : Consider Damain Adaptation Training, i.e. DANN; 
                              False : Consider Source-Only Training
        show_accs           : Calculate and Plot (as Table) the Classification Accuracies
        plot_domain_disc    : Calculate and Plot (as Table) Accuracies when Discriminating the Domains
        plot_entangle       : Calculate and Plot Manifold Entanglement Metric
        dim_red             : Plot a 2D Visualization of the Data/Representations as Scatter and Gaussian KDE(only for t-SNE)!
        IF True: tSNE_plot, pca_plot, isomap_plot, LLE_plot: The kinds of low Dimensional Representations
                                -> t-SNE yields a nice Viz
                                -> PCA, Isomap and LLE(Local Lin Embedding) dont yield good results
        TODO: Make the code readable...
        TODO: Include better plot for disentanglement used in MA
    '''
    I_source = [3]#[1,2,3,4,5]
    I_target = [4]#[0,1,2,3,4,5]
    accs = np.zeros((6,6))#(len(I_source),len(I_target)))
    B = [6,7,8,10,12,14,16,18,20,22,24]
    b = B[depth]
    
    accs_discr = np.zeros((len(I_source),len(I_target)))
    d = np.zeros((6,6,10,10))
    d_t = np.zeros((6,6,10,10))
    per = np.ones((6,6,10,10))
    per_t = np.ones((6,6,10,10))
    snr_s = 3
    C = [['xkcd:forest green','xkcd:blue','xkcd:turquoise','xkcd:violet','xkcd:green','xkcd:mustard','xkcd:blue green','xkcd:orange','xkcd:hot pink','xkcd:cherry red'],['xkcd:olive','xkcd:light blue','xkcd:aqua','xkcd:light purple','xkcd:pale green','xkcd:beige','xkcd:grey blue','xkcd:light orange','xkcd:light pink','xkcd:pinkish red']]
    for s in I_source:
        for t in I_target:
            #t = 0
            if s== t:#No DA
                mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = data_create(s, s, snr = snr_s, plotsamples = False)
                Cha = [1,1,3,3,3,1]
                if DA == True:
                    channel = Cha[s]
                else:
                    channel = 3
                    if mnist_train_x.shape[3] < channel:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                mnist_train_x   = mnist_train_x.astype('float32')
                mnist_test_x    = mnist_test_x.astype('float32')
                mnist_train_y_dec = mnist_train_y
                mnist_test_y_dec  = mnist_test_y
                mnist_train_y = to_categorical(mnist_train_y)
                mnist_test_y = to_categorical(mnist_test_y)
                classifier, accs[s,t]=load_model_from_data(s,t,depth,snr_s,channel,DA = DA)
                classifier.build(input_shape = (1,28,28,channel))
                #classifier.summary()
                if plot_entangle == True:
                    for i in range(10):
                        d[s,t,i,:],per[s,t,i,:] = dann_manifold_disentanglement(classifier, mnist_train_x  , mnist_train_y_dec, 15, i,depth)
                if dim_red == True:
                    discr_train_x = mnist_train_x[:10000,:,:,:]
                    YYY     = mnist_train_y_dec[:10000]
                    pca_50 = PCA(n_components=50)
                    discr_train_x = discr_train_x.reshape(-1,28*28*channel)
                    pca_result_50 = pca_50.fit_transform(discr_train_x)
                    x   = pca_result_50[:, 0]
                    x1  = pca_result_50[:10000, 0]
                    y1 = pca_result_50[:10000, 1]
                    if pca_plot == True:
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)#,edgecolors='gray')
                        plt.title('PCA Unprocessed Data - var='+str(var)+'%, Domain '+str(s))
                        plt.axis('off')
                        plt.show()
                    if tSNE_plot == True:
                        tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                        X_transformed = tsne.fit_transform(pca_result_50)
                        # Plot t-SNE outputs
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                        plt.title('t-SNE Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        #ax.legend()
                        plt.show()
                    if isomap_plot == True:
                        isomap = Isomap(n_components=2)
                        X_transformed = isomap.fit_transform(pca_result_50)
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                        plt.title('Isomap Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        plt.show()    
                    if LLE_plot == True:
                        lle = LocallyLinearEmbedding(n_components=2)
                        X_transformed = lle.fit_transform(pca_result_50)
                        x1 = X_transformed[:10000, 0]
                        y1 = X_transformed[:10000, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                        plt.title('LLE Unprocessed Data, Domain '+str(s))
                        plt.axis('off')
                        plt.show()    
                        
                    for i in range(depth):
                        #funce = selfmade_backend(x,classifier,B[i])#K.function([classifier.layers[0].input], [classifier.layers[B[i]].output])
                        representation = selfmade_backend(discr_train_x,classifier,B[i],DA)#funce(discr_train_x)
                        representation = representation.numpy()
                        rep_plot       = representation[:10000,:]#Random
                        pca_50 = PCA(n_components=50)
                        pca_result_50 = pca_50.fit_transform(rep_plot)
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        
                        if pca_plot == True:
                            x1 = pca_result_50[:10000, 0]
                            y1 = pca_result_50[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('PCA Dense Layer '+str(i)+' - var='+str(var)+'%, Domain '+str(s))
                            #plt.legend(YYY, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
                            plt.axis('off')
                            plt.show()
                        if tSNE_plot == True:
                            tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                            X_transformed = tsne.fit_transform(pca_result_50)
                            # Plot t-SNE outputs
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('t-SNE Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.show()
                        if isomap_plot == True:
                            isomap = Isomap(n_components=2)
                            X_transformed = isomap.fit_transform(pca_result_50)
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('Isomap Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.show()    
                        if LLE_plot == True:
                            lle = LocallyLinearEmbedding(n_components=2)
                            X_transformed = lle.fit_transform(pca_result_50)
                            x1 = X_transformed[:10000, 0]
                            y1 = X_transformed[:10000, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            plt.title('LLE  Dense Layer '+str(i)+', Domain '+str(s))
                            plt.axis('off')
                            plt.show()        
            else:
                #DA=True
                mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = data_create(s, s, snr = snr_s, plotsamples = False)
                mnist_m_train_x, mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(t, t, snr = snr_s, plotsamples = False)
                Cha = [1,1,3,3,3,1]
                if DA == True:
                    print('ha')
                    channel = max(Cha[s],Cha[t])
                    if Cha[s] < Cha[t]:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                    elif Cha[s] > Cha[t]:
                        mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                        mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
                else:
                    channel = 3
                    if mnist_train_x.shape[3] < channel:
                        mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
                        mnist_test_x = np.repeat(mnist_test_x, 3, axis=3)
                    if mnist_m_train_x.shape[3] < channel:
                        mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
                        mnist_m_test_x = np.repeat(mnist_m_test_x, 3, axis=3)
                #inputsize = 100    #mnist_test_x[1]*mnist_test_x[2]*mnist_test_x[3]
                mnist_train_x   = mnist_train_x.astype('float32')
                mnist_test_x    = mnist_test_x.astype('float32')
                mnist_m_train_x = mnist_m_train_x.astype('float32')
                mnist_m_test_x  = mnist_m_test_x.astype('float32')
                mnist_train_y_dec = mnist_train_y
                mnist_test_y_dec  = mnist_test_y
                mnist_train_y = to_categorical(mnist_train_y)
                mnist_test_y = to_categorical(mnist_test_y)
                mnist_m_train_y_dec = mnist_m_train_y
                mnist_m_test_y_dec  = mnist_m_test_y
                mnist_m_train_y = to_categorical(mnist_m_train_y)
                mnist_m_test_y = to_categorical(mnist_m_test_y)

                #load model
                classifier, accs[s,t]=load_model_from_data(s,t,depth,snr_s,channel = channel,DA = DA)
                classifier.build(input_shape = (1,28,28,channel))
                #classifier.summary()
                print(classifier.layers[b].name)
                #func = selfmade_backend(x,classifier,b)
                
                #calc domain discrimination
                if plot_domain_disc == True:
                    discriminator =  classify_domain(100)
                    discr_train_x   = np.concatenate([mnist_train_x[:20000,:,:,:],mnist_m_train_x[30000:50000,:,:,:]])
                    discr_test_x    = np.concatenate([mnist_test_x[:3000,:,:,:],mnist_m_test_x[4000:7000,:,:,:]])
                    discr_train_y   = np.concatenate([[0 for i in range(20000)],[1 for i in range(20000)]])
                    discr_test_y    = np.concatenate([[0 for i in range(3000)],[1 for i in range(3000)]])
                    discr_train_x, discr_train_y = shuffle(discr_train_x, discr_train_y, random_state=0)
                    discr_test_x, discr_test_y = shuffle(discr_test_x, discr_test_y, random_state=0)
                    discr_train_y   = to_categorical(discr_train_y)
                    discr_test_y    = to_categorical(discr_test_y)
                    
                    rep_train = selfmade_backend(discr_train_x,classifier,b,DA)#func(discr_train_x)
                    rep_train = rep_train.numpy()
                    rep_test = selfmade_backend(discr_test_x,classifier,b,DA)#func(discr_test_x)
                    rep_test = rep_test.numpy()                
                    discriminator.compile(loss='categorical_crossentropy', 
                              optimizer='adam',
                              metrics=['accuracy'])
                    discriminator.fit(rep_train, discr_train_y, epochs=10, batch_size=16)
                    loss, acc = discriminator.evaluate(rep_test,discr_test_y,batch_size=8)
                    accs_discr[s,t] = acc
                    
                
                #entanglement
                if plot_entangle == True:
                    for i in range(10):
                        d[s,t,i,:],per[s,t,i,:]      = dann_manifold_disentanglement(classifier, mnist_train_x  , mnist_train_y_dec, 15, i,depth)
                        d_t[s,t,i,:],per_t[s,t,i,:]  = dann_manifold_disentanglement(classifier, mnist_m_train_x, mnist_m_train_y_dec, 15, i,depth)
                    
                #pca, t-sne, isomap, lle 
                if dim_red == True:
                    discr_train_x   = np.concatenate([mnist_train_x[:20000,:,:,:],mnist_m_train_x[30000:50000,:,:,:]])
                    #discr_train_lab = np.concatenate([mnist_train_y[:20000,:],mnist_m_train_y[30000:50000,:]])
                    YYY     = mnist_train_y_dec[:20000]
                    YYYY    = mnist_m_train_y_dec[30000:50000]
                    pca_50 = PCA(n_components=50)
                    discr_train_x = discr_train_x.reshape(-1,28*28*channel)
                    pca_result_50 = pca_50.fit_transform(discr_train_x)
                    x   = pca_result_50[:, 0]
                    x1  = pca_result_50[:20000, 0]
                    x2  = pca_result_50[20000:, 0]
                    y1 = pca_result_50[:20000, 1]
                    y2 = pca_result_50[20000:, 1]
                    if pca_plot == True:
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
    
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)#,edgecolors='gray')
                            scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)#,edgecolors='gray')
                        plt.title('PCA Unprocessed Data - var='+str(var)+'%, Source '+str(s)+', Target '+str(t))
                        plt.axis('off')
                        plt.show()
                    if tSNE_plot == True:
                        tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                        X_transformed = tsne.fit_transform(pca_result_50)
                        # Plot t-SNE outputs
                        x1 = X_transformed[:20000, 0]
                        x2 = X_transformed[20000:, 0]
                        y1 = X_transformed[:20000, 1]
                        y2 = X_transformed[20000:, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                        plt.title('t-SNE Unprocessed Data, Source '+str(s)+', Target '+str(t))
                        plt.axis('off')
                        #ax.legend()
                        plt.show()
                    if isomap_plot == True:
                        isomap = Isomap(n_components=2)
                        X_transformed = isomap.fit_transform(pca_result_50)
                        x1 = X_transformed[:20000, 0]
                        x2 = X_transformed[20000:, 0]
                        y1 = X_transformed[:20000, 1]
                        y2 = X_transformed[20000:, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            scatter = ax.scatter(x2[YYYY == ii], y2[YYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                        plt.title('Isomap Unprocessed Data, Source '+str(s)+', Target '+str(t))
                        plt.axis('off')
                        plt.show()    
                    if LLE_plot == True:
                        lle = LocallyLinearEmbedding(n_components=2)
                        X_transformed = lle.fit_transform(pca_result_50)
                        x1 = X_transformed[:20000, 0]
                        x2 = X_transformed[20000:, 0]
                        y1 = X_transformed[:20000, 1]
                        y2 = X_transformed[20000:, 1]
                        fig, ax = plt.subplots()
                        for ii in range(10):
                            scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                            scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                        plt.title('LLE Unprocessed Data, Source '+str(s)+', Target '+str(t))
                        plt.axis('off')
                        plt.show()    
                        
                    for i in range(depth):
                        print(i)
                        #funce = selfmade_backend(x,classifier,B[i])#K.function([classifier.layers[0].input], [classifier.layers[B[i]].output])
                        representation = selfmade_backend(discr_train_x,classifier,B[i],DA)#funce(discr_train_x)
                        representation = representation.numpy()
                        rep_plot       = representation[:40000,:]#Random
                        print("mid")
                        pca_50 = PCA(n_components=50)
                        pca_result_50 = pca_50.fit_transform(rep_plot)
                        var  = pca_50.explained_variance_ratio_[0]+pca_50.explained_variance_ratio_[1]
                        print(i)
                        if pca_plot == True:
                            x1 = pca_result_50[:20000, 0]
                            x2 = pca_result_50[20000:, 0]
                            y1 = pca_result_50[:20000, 1]
                            y2 = pca_result_50[20000:, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                                scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                            plt.title('PCA Dense Layer '+str(i)+' - var='+str(var)+'%, Source '+str(s)+', Target '+str(t))
                            #plt.legend(YYY, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
                            plt.axis('off')
                            plt.show()
                        if tSNE_plot == True:
                            tsne = TSNE(n_components=2,perplexity=60.0,init='pca')
                            X_transformed = tsne.fit_transform(pca_result_50)
                            # Plot t-SNE outputs
                            x1 = X_transformed[:20000, 0]
                            x2 = X_transformed[20000:, 0]
                            y1 = X_transformed[:20000, 1]
                            y2 = X_transformed[20000:, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                                scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                            plt.title('t-SNE Dense Layer '+str(i)+', Source '+str(s)+', Target '+str(t))
                            #ax.legend()
                            plt.axis('off')
                            plt.show()
                        if isomap_plot == True:
                            isomap = Isomap(n_components=2)
                            print("in")
                            X_transformed = isomap.fit_transform(discr_train_x)
                            print("out")
                            x1 = X_transformed[:20000, 0]
                            x2 = X_transformed[20000:, 0]
                            y1 = X_transformed[:20000, 1]
                            y2 = X_transformed[20000:, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                                scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                            plt.title('Isomap Dense Layer'+str(i)+', Source '+str(s)+', Target '+str(t))
                            plt.axis('off')
                            plt.show()    
                        if LLE_plot == True:
                            lle = LocallyLinearEmbedding(n_components=2)
                            X_transformed = lle.fit_transform(discr_train_x)
                            x1 = X_transformed[:20000, 0]
                            x2 = X_transformed[20000:, 0]
                            y1 = X_transformed[:20000, 1]
                            y2 = X_transformed[20000:, 1]
                            fig, ax = plt.subplots()
                            for ii in range(10):
                                scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
                                scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
                            plt.title('LLE  Dense Layer'+str(i)+', Source '+str(s)+', Target '+str(t))
                            plt.axis('off')
                            plt.show()        
                            
        #plot domain discr
    if plot_domain_disc == True:
                    DD_DANN    = np.array(np.matrix('.5 .5 .5 .5 .62 .72; .5 .5 .5 .54 .64 .73; .5 .5 .5 .57 .6 .7; .5 .5 .5 .5 .5 .68; .53 .5 .54 .5 .5 .72; .63 .66 .64 .78 .75 .5'))
                    DD_SOUR    = np.array(np.matrix('0.5 .67 .5 .98 .98 .84; .62 .5 .5 .98 .98 .87; .5 .53 .5 .98 .99 .83; .82 .61 .5 .5 .56 .83; .65 .66 .84 .58 .5 .85; .89 .93 .91 .99 .99 .5'))
                    plt.rcParams['figure.dpi']=1000
        
                    title_text = 'Domain Discrimination Accuracy'
                    #footer_text = 'June 24, 2020'
                    data    = DD_DANN#accs_discr
                    
                   
                    columns = ('MNIST', 'MNIST-n GS', 'MNIST-n RGB', 'MNIST-c', 'MNIST-l','MNIST-r')
                    DDisc_ACC = LinearSegmentedColormap.from_list('DDisc_ACC', (
                    # Edit this gradient at https://eltos.github.io/gradient/#DDisc_ACC=50:FFFFFF-85:FF5760
                    (0.000, (1.000, 1.000, 1.000)),
                    (0.500, (1.000, 1.000, 1.000)),
                    (0.850, (1.000, 0.341, 0.376)),
                    (1.000, (1.000, 0.341, 0.376))))
                    #rows    = columns
                    colours = DDisc_ACC(data)#accs_discr)#plt.cm.hot(accs)
                    cell_text = []
                    for row in data:
                        cell_text.append([f'{x:1.2f}' for x in row])
                    # Get some lists of color specs for row and column headers
                    rcolors = plt.cm.BuPu(np.full(len(columns), 0.1))
                    ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))
                    # Create the figure. Setting a small pad on tight_layout
                    # seems to better regulate white space. Sometimes experimenting
                    # with an explicit figsize here can produce better outcome.
                    plt.figure(linewidth=2,
                               tight_layout={'pad':1},
                               figsize=(6,3)
                              )
                    
                    # Add a table at the bottom of the axes
                    the_table = plt.table(cellText=cell_text,
                                          cellColours = colours,
                                          rowLabels=columns,
                                          rowColours=rcolors,
                                          rowLoc='right',
                                          colWidths=[0.3] * 6,
                                          colColours=ccolors,
                                          colLabels=columns,
                                          loc='center')
                    the_table.auto_set_font_size(False)
                    the_table.set_fontsize(10)
                    the_table.scale(1, 1.5)
                    ax = plt.gca()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    plt.box(on=None)
                    # Add title
                    plt.suptitle(title_text, y=0.9)
                    plt.draw()
                    fig = plt.gcf()
                    plt.savefig('Domain_discrAccuracys_DA.pdf',
                                bbox_inches='tight',#bbox='tight',
                                edgecolor=fig.get_edgecolor(),
                                facecolor=fig.get_facecolor(),
                                dpi=700
                                )
        
        #plot_acc_table
    if show_accs == True:
            DA_DANN     = np.array(np.matrix('0.99 .99 .99 .97 .93 .65; 0.99 .99 .99 .96 .89 .59; 0.99 .99 .99 .97 .91 .55; 0.98 .98 .98 .98 .97 .48; 0.98 .98 .95 .98 .98 .52; 0.82 .76 .79 .18 .17 .97'))
            DA_SOUR     = np.array(np.matrix('0.99 .99 .99 .18 .17 .65; 0.99 .99 .99 .19 .19 .57; 0.99 .99 .99 .2  .18 .64; 0.99 .96 .95 .98 .97 .52;0.98 .96 .89 .98 .98 .45;0.49 .5  .52 .15 .17 .99'))
            data    = accs
            #data = DA_SOUR
           
            DA_ACC = LinearSegmentedColormap.from_list('DA_ACC', (
            # Edit this gradient at https://eltos.github.io/gradient/#DA_ACC=0:F6353F-85:F9812D-100:8DE87A
            (0.000, (0.965, 0.208, 0.247)),
            (0.850, (0.976, 0.506, 0.176)),
            (1.000, (0.553, 0.910, 0.478))))
            colours = DA_ACC(data)#plt.cm.hot(accs)
            title_text = 'Classification Accuracy'
            #footer_text = 'June 24, 2020'
            
            columns = ('MNIST', 'MNIST-n GS', 'MNIST-n RGB', 'MNIST-c', 'MNIST-l','MNIST-r')
            #rows    = columns
            cell_text = []
            for row in data:
                cell_text.append([f'{x:1.2f}' for x in row])
            # Get some lists of color specs for row and column headers
            rcolors = plt.cm.BuPu(np.full(len(columns), 0.1))
            ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))
            plt.figure(linewidth=2,
                       tight_layout={'pad':1},
                       figsize=(6,3)
                       #figsize=(5,3)
                      )
            # Add a table at the bottom of the axes
            the_table = plt.table(cellText=cell_text,
                                  cellColours = colours,
                                  rowLabels=columns,
                                  rowColours=rcolors,
                                  rowLoc='right',
                                  colWidths=[0.3] * 6,
                                  colColours=ccolors,
                                  colLabels=columns,
                                  loc='center')
            for k, cell in the_table._cells.items():
                cell.set_edgecolor('black')                    
                cell.set_alpha(0.5)
            # Scaling is the only influence we have over top and bottom cell padding.
            # Make the rows taller (i.e., make cell y scale larger).
            the_table.scale(1, 1.5)
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            # Hide axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # Hide axes border
            plt.box(on=None)
            # Add title
            plt.suptitle(title_text, y=0.9)
            plt.draw()
            fig = plt.gcf()
            plt.savefig('Accuracys_S.pdf',
                        bbox_inches='tight',
                        #bbox='tight',
                        edgecolor=fig.get_edgecolor(),
                        facecolor=fig.get_facecolor(),
                        dpi=700)
    #plot entangle 
    if plot_entangle == True:
           X = [i for i in range(1,depth+2)]
           doms = ['MNIST', 'n-MNIST', 'n-MNIST-3D', 'c-MNIST', 'l-MNIST','r-MNIST']
           #d_t = d + 1
           for labl in range(10):
              # for s in range(6):
                   fig, axs = plt.subplots(2, 3)
                   fig.suptitle('Disentanglement of class '+str(labl))
                   for t in range(3):
                       axs[0, t].plot(X, d[s,t,labl,:5],label='Source Domain')
                       if s != t:
                           axs[0, t].plot(X, d_t[s,t,labl,:5],label='Target Domain')
                       #axs[s, t].set_xlabel('Dense Layer')
                       axs[0,t].set_xticks(X)
                       axs[0, t].set_ylabel('Disentanglement')
                       axs[0, t].set_title(doms[s] + " to " + doms[t])
                   for t in range(3,6):
  
                       axs[1, t-3].plot(X, d[s,t,labl,:5],label='Source Domain')
                       if s!= t:
                           axs[1, t-3].plot(X, d_t[s,t,labl,:5],label='Target Domain')
                       #axs[s, t].set_xlabel('Dense Layer')
                       axs[1, t-3].set_ylabel('Disentanglement')
                       axs[1,t-3].set_xticks(X)
                       axs[1, t-3].set_title(doms[s] + " to " + doms[t])
                   fig.tight_layout()
                   plt.show()
                     
    return None

'''TODO : Rewrite specific calculation and boxplots for class entanglement values along networks 
            !!!!!!!!!!!!AND Vectorize the Code!!!!!!!!!!!!!!'''
# =============================================================================
# =============================================================================
# # from tensorflow.python.ops.numpy_ops import np_config
# # np_config.enable_numpy_behavior()
# # print('gogogog')
# # depth = 4
# # mnist_train_x,mnist_train_y, mnist_test_x, mnist_test_y = data_create(0, 0)
# # mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
# # mnist_train_x       = mnist_train_x.astype('float32')
# # mnist_train_x_arr   = mnist_train_x.reshape(-1,28*28*3)
# # mnist_train_y_dec   = mnist_train_y
# # mnist_train_y       = to_categorical(mnist_train_y)
# # 
# # mnist_m_train_x,mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(3, 3)
# # mnist_m_train_x     = mnist_m_train_x.astype('float32')
# # mnist_m_train_x_arr = mnist_m_train_x.reshape(-1,28*28*3)
# # mnist_m_train_y_dec = mnist_m_train_y
# # mnist_m_train_y     = to_categorical(mnist_m_train_y)
# # I           = [784*3,4608*2,4608,4608,4608,2048,512,128,64,64,64,64,64,64,64]
# # B           = [0,4,6,8,10,11,14,17,20,23,26,29,32]
# # B_10        = [0,3,5,7,9,11,14,17,20,23,26,29,32]
# # #d_03s= np.zeros((7,10))
# # #dt_03s= np.zeros((7,10))
# # d_30s       = np.zeros((7,10))
# # dt_30s      = np.zeros((7,10))
# # d_03d       = np.zeros((7,10))
# # dt_03d      = np.zeros((7,10))
# # d_30d       = np.zeros((7,10))
# # dt_30d      = np.zeros((7,10))
# # 
# # d_53s       = np.zeros((7,10))
# # dt_53s      = np.zeros((7,10))
# # d_53d       = np.zeros((7,10))
# # dt_53d      = np.zeros((7,10))
# # # =============================================================================
# # # 
# # # model, _=load_model_from_data(0,3,4,snr_s = 3,channel=3,DA = False)
# # # model.build(input_shape = (1,28,28,3))
# # # model.summary()
# # # rep_train = mnist_train_x_arr
# # # rep_test  = mnist_m_train_x_arr
# # # for i in range(10):
# # #     d_03s[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = model.layers[0](mnist_train_x)
# # # rep_test  = model.layers[0](mnist_m_train_x)
# # # rep_train = model.layers[1](rep_train)
# # # rep_test  = model.layers[1](rep_test)
# # # rep_train = model.layers[2](rep_train)
# # # rep_test  = model.layers[2](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = model.layers[3](rep_train)
# # # rep_test  = model.layers[3](rep_test)
# # # rep_train = model.layers[4](rep_train)
# # # rep_test  = model.layers[4](rep_test)
# # # rep_train = model.layers[5](rep_train)
# # # rep_test  = model.layers[5](rep_test)
# # # rep_train = model.layers[6](rep_train)
# # # rep_test  = model.layers[6](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # # rep_train = model.layers[7](rep_train)
# # # rep_test  = model.layers[7](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = model.layers[8](rep_train)
# # # rep_test  = model.layers[8](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = model.layers[9](rep_train)
# # # rep_test  = model.layers[9](rep_test)
# # # rep_train = model.layers[10](rep_train)
# # # rep_test  = model.layers[10](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # rep_train = model.layers[11](rep_train)
# # # rep_test  = model.layers[11](rep_test)
# # # rep_train = model.layers[12](rep_train)
# # # rep_test  = model.layers[12](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03s[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03s[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03s', d_03s)
# # #     np.save('d03ts',dt_03s)
# # # print('fin 03s')
# # # 
# # # =============================================================================
# # # =============================================================================
# # # 
# # # model, _=load_model_from_data(0,3,4,snr_s = 3,channel=3,DA = True)
# # # model.build(input_shape = (1,28,28,3))
# # # model.summary()
# # # rep_train = mnist_train_x_arr
# # # rep_test  = mnist_m_train_x_arr
# # # for i in range(10):
# # #     d_03d[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = model.layers[0](mnist_train_x)
# # # rep_test  = model.layers[0](mnist_m_train_x)
# # # rep_train = model.layers[1](rep_train)
# # # rep_test  = model.layers[1](rep_test)
# # # rep_train = model.layers[2](rep_train)
# # # rep_test  = model.layers[2](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = model.layers[3](rep_train)
# # # rep_test  = model.layers[3](rep_test)
# # # rep_train = model.layers[4](rep_train)
# # # rep_test  = model.layers[4](rep_test)
# # # rep_train = model.layers[5](rep_train)
# # # rep_test  = model.layers[5](rep_test)
# # # rep_train = model.layers[6](rep_train)
# # # rep_test  = model.layers[6](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # # rep_train = model.layers[7](rep_train)
# # # rep_test  = model.layers[7](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = model.layers[8](rep_train)
# # # rep_test  = model.layers[8](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = model.layers[9](rep_train)
# # # rep_test  = model.layers[9](rep_test)
# # # rep_train = model.layers[10](rep_train)
# # # rep_test  = model.layers[10](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # rep_train = model.layers[11](rep_train)
# # # rep_test  = model.layers[11](rep_test)
# # # rep_train = model.layers[12](rep_train)
# # # rep_test  = model.layers[12](rep_test)
# # # print('start')
# # # for i in range(10):
# # #     d_03d[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# # #     dt_03d[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# # #     np.save('d03d', d_03d)
# # #     np.save('d03td',dt_03d)
# # # print('fin 03d')
# # # 
# # # =============================================================================
# # mnist_train_x,mnist_train_y, mnist_test_x, mnist_test_y = data_create(3, 3)
# # #mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
# # mnist_train_x       = mnist_train_x.astype('float32')
# # mnist_train_x_arr   = mnist_train_x.reshape(-1,28*28*3)
# # mnist_train_y_dec   = mnist_train_y
# # mnist_train_y       = to_categorical(mnist_train_y)
# # 
# # mnist_m_train_x,mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(0, 0)
# # mnist_m_train_x = np.repeat(mnist_m_train_x, 3, axis=3)
# # mnist_m_train_x     = mnist_m_train_x.astype('float32')
# # mnist_m_train_x_arr = mnist_m_train_x.reshape(-1,28*28*3)
# # mnist_m_train_y_dec = mnist_m_train_y
# # mnist_m_train_y     = to_categorical(mnist_m_train_y)
# # 
# # 
# # model, _=load_model_from_data(3,0,4,snr_s = 3,channel=3,DA = False)
# # model.build(input_shape = (1,28,28,3))
# # model.summary()
# # rep_train = mnist_train_x_arr
# # rep_test  = mnist_m_train_x_arr
# # for i in range(10):
# #     d_30s[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# #     dt_30s[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = model.layers[0](mnist_train_x)
# # rep_test  = model.layers[0](mnist_m_train_x)
# # rep_train = model.layers[1](rep_train)
# # rep_test  = model.layers[1](rep_test)
# # rep_train = model.layers[2](rep_train)
# # rep_test  = model.layers[2](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = model.layers[3](rep_train)
# # rep_test  = model.layers[3](rep_test)
# # rep_train = model.layers[4](rep_train)
# # rep_test  = model.layers[4](rep_test)
# # rep_train = model.layers[5](rep_train)
# # rep_test  = model.layers[5](rep_test)
# # rep_train = model.layers[6](rep_train)
# # rep_test  = model.layers[6](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # rep_train = model.layers[7](rep_train)
# # rep_test  = model.layers[7](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = model.layers[8](rep_train)
# # rep_test  = model.layers[8](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = model.layers[9](rep_train)
# # rep_test  = model.layers[9](rep_test)
# # rep_train = model.layers[10](rep_train)
# # rep_test  = model.layers[10](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # rep_train = model.layers[11](rep_train)
# # rep_test  = model.layers[11](rep_test)
# # rep_train = model.layers[12](rep_train)
# # rep_test  = model.layers[12](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30s[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30s[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30s', d_30s)
# #     np.save('d30ts',dt_30s)
# # print('fin 30s')
# # 
# # 
# # model, _=load_model_from_data(3,0,4,snr_s = 3,channel=3,DA = True)
# # model.build(input_shape = (1,28,28,3))
# # model.summary()
# # rep_train = mnist_train_x_arr
# # rep_test  = mnist_m_train_x_arr
# # for i in range(10):
# #     d_30d[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# #     dt_30d[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = model.layers[0](mnist_train_x)
# # rep_test  = model.layers[0](mnist_m_train_x)
# # rep_train = model.layers[1](rep_train)
# # rep_test  = model.layers[1](rep_test)
# # rep_train = model.layers[2](rep_train)
# # rep_test  = model.layers[2](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = model.layers[3](rep_train)
# # rep_test  = model.layers[3](rep_test)
# # rep_train = model.layers[4](rep_train)
# # rep_test  = model.layers[4](rep_test)
# # rep_train = model.layers[5](rep_train)
# # rep_test  = model.layers[5](rep_test)
# # rep_train = model.layers[6](rep_train)
# # rep_test  = model.layers[6](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # rep_train = model.layers[7](rep_train)
# # rep_test  = model.layers[7](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = model.layers[8](rep_train)
# # rep_test  = model.layers[8](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = model.layers[9](rep_train)
# # rep_test  = model.layers[9](rep_test)
# # rep_train = model.layers[10](rep_train)
# # rep_test  = model.layers[10](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # rep_train = model.layers[11](rep_train)
# # rep_test  = model.layers[11](rep_test)
# # rep_train = model.layers[12](rep_train)
# # rep_test  = model.layers[12](rep_test)
# # print('start')
# # for i in range(10):
# #     d_30d[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_30d[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d30d', d_30d)
# #     np.save('d30td',dt_30d)
# # print('fin 30d')
# # 
# # mnist_train_x,mnist_train_y, mnist_test_x, mnist_test_y = data_create(5, 5)
# # mnist_train_x = np.repeat(mnist_train_x, 3, axis=3)
# # mnist_train_x       = mnist_train_x.astype('float32')
# # mnist_train_x_arr   = mnist_train_x.reshape(-1,28*28*3)
# # mnist_train_y_dec   = mnist_train_y
# # mnist_train_y       = to_categorical(mnist_train_y)
# # 
# # mnist_m_train_x,mnist_m_train_y, mnist_m_test_x, mnist_m_test_y = data_create(3, 3)
# # mnist_m_train_x     = mnist_m_train_x.astype('float32')
# # mnist_m_train_x_arr = mnist_m_train_x.reshape(-1,28*28*3)
# # mnist_m_train_y_dec = mnist_m_train_y
# # mnist_m_train_y     = to_categorical(mnist_m_train_y)
# # 
# # model, _=load_model_from_data(5,3,4,snr_s = 3,channel=3,DA = False)
# # model.build(input_shape = (1,28,28,3))
# # model.summary()
# # rep_train = mnist_train_x_arr
# # rep_test  = mnist_m_train_x_arr
# # for i in range(10):
# #     d_53s[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# #     dt_53s[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = model.layers[0](mnist_train_x)
# # rep_test  = model.layers[0](mnist_m_train_x)
# # rep_train = model.layers[1](rep_train)
# # rep_test  = model.layers[1](rep_test)
# # rep_train = model.layers[2](rep_train)
# # rep_test  = model.layers[2](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = model.layers[3](rep_train)
# # rep_test  = model.layers[3](rep_test)
# # rep_train = model.layers[4](rep_train)
# # rep_test  = model.layers[4](rep_test)
# # rep_train = model.layers[5](rep_train)
# # rep_test  = model.layers[5](rep_test)
# # rep_train = model.layers[6](rep_train)
# # rep_test  = model.layers[6](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # rep_train = model.layers[7](rep_train)
# # rep_test  = model.layers[7](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = model.layers[8](rep_train)
# # rep_test  = model.layers[8](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = model.layers[9](rep_train)
# # rep_test  = model.layers[9](rep_test)
# # rep_train = model.layers[10](rep_train)
# # rep_test  = model.layers[10](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # rep_train = model.layers[11](rep_train)
# # rep_test  = model.layers[11](rep_test)
# # rep_train = model.layers[12](rep_train)
# # rep_test  = model.layers[12](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53s[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53s[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53s', d_53s)
# #     np.save('d53ts',dt_53s)
# # print('fin 53s')
# # 
# # 
# # model, _=load_model_from_data(5,3,4,snr_s = 3,channel=3,DA = True)
# # model.build(input_shape = (1,28,28,3))
# # model.summary()
# # rep_train = mnist_train_x_arr
# # rep_test  = mnist_m_train_x_arr
# # for i in range(10):
# #     d_53d[0,i],_      = manifold_disentanglements2(rep_train, mnist_train_y_dec, 15, i,depth)
# #     dt_53d[0,i],_     = manifold_disentanglements2(rep_test, mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = model.layers[0](mnist_train_x)
# # rep_test  = model.layers[0](mnist_m_train_x)
# # rep_train = model.layers[1](rep_train)
# # rep_test  = model.layers[1](rep_test)
# # rep_train = model.layers[2](rep_train)
# # rep_test  = model.layers[2](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[1,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[1,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = model.layers[3](rep_train)
# # rep_test  = model.layers[3](rep_test)
# # rep_train = model.layers[4](rep_train)
# # rep_test  = model.layers[4](rep_test)
# # rep_train = model.layers[5](rep_train)
# # rep_test  = model.layers[5](rep_test)
# # rep_train = model.layers[6](rep_train)
# # rep_test  = model.layers[6](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[2,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[2,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = tf.reshape(rep_train, [-1, 4 * 4 * 64])
# # rep_test = tf.reshape(rep_test, [-1, 4 * 4 * 64])
# # rep_train = model.layers[7](rep_train)
# # rep_test  = model.layers[7](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[3,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[3,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = model.layers[8](rep_train)
# # rep_test  = model.layers[8](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[5,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[5,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = model.layers[9](rep_train)
# # rep_test  = model.layers[9](rep_test)
# # rep_train = model.layers[10](rep_train)
# # rep_test  = model.layers[10](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[6,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[6,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # rep_train = model.layers[11](rep_train)
# # rep_test  = model.layers[11](rep_test)
# # rep_train = model.layers[12](rep_train)
# # rep_test  = model.layers[12](rep_test)
# # print('start')
# # for i in range(10):
# #     d_53d[4,i],_   = manifold_disentanglements2(rep_train.reshape(60000,-1), mnist_train_y_dec, 15, i,depth)
# #     dt_53d[4,i],_  = manifold_disentanglements2(rep_test.reshape(60000,-1), mnist_m_train_y_dec, 15, i,depth)
# #     np.save('d53d', d_53d)
# #     np.save('d53td',dt_53d)
# # print('fin 53d')
# # # =============================================================================
# # # x = self.feature_extractor_layer0(x)
# # # x = self.feature_extractor_layer1(x, training=train)
# # # x = self.feature_extractor_layer2(x)
# # # 
# # # x = self.feature_extractor_layer3(x)
# # # x = self.feature_extractor_layer4(x, training=train)
# # # x = self.feature_extractor_layer5(x, training=train)
# # # x = self.feature_extractor_layer6(x)
# # #  
# # #  feature = tf.reshape(x, [-1, 4 * 4 * 64])
# # # 
# # #     x = x.reshape(x.shape[0],28,28,-1)
# # #     for i in range(ind+1):
# # #         x = model.layers[i](x)
# # #         if i == 6:
# # #             x = tf.reshape(x, [-1, 4 * 4 * 64])
# # #             #x = x
# # #     return x
# # # =============================================================================
# # 
# # 
# # 
# # 
# # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# # from matplotlib.cbook import get_sample_data
# # 
# # def main(idxy, data, dimred):
# #     x = dimred[idxy,0]
# #     y = dimred[idxy,1]
# #     image = dimred[data,:]
# #     fig, ax = plt.subplots()
# #     isomap = Isomap(n_components=2)
# # # =============================================================================
# # #     X_transformed = isomap.fit_transform(discr_train_x)
# # #     x1 = X_transformed[:20000, 0]
# # #     x2 = X_transformed[20000:, 0]
# # #     y1 = X_transformed[:20000, 1]
# # #     y2 = X_transformed[20000:, 1]
# # #     fig, ax = plt.subplots()
# # #     for ii in range(10):
# # #         scatter = ax.scatter(x1[YYY == ii], y1[YYY == ii], c=C[0][ii],s = 3,alpha = 0.75)
# # #         scatter = ax.scatter(x2[YYYY == ii], y2[YYYY == ii], c=C[1][ii],s = 3,alpha = 0.75)
# # #     plt.title('Isomap Dense Layer'+str(i)+', Source '+str(s)+', Target '+str(t))
# # #     plt.axis('off')
# # #     plt.show()
# # # =============================================================================
# #     imscatter(x, y, image_path, zoom=0.1, ax=ax)
# #     ax.plot(x, y)
# #     plt.show()
# # 
# # def imscatter(x, y, image, ax=None, zoom=1):
# # 
# #     try:
# #         image = plt.imread(image)
# #     except TypeError:
# #         # Likely already an array...
# #         pass
# #     im = OffsetImage(image, zoom=zoom)
# #     x, y = np.atleast_1d(x, y)
# #     artists = []
# #     for x0, y0 in zip(x, y):
# #         ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
# #         artists.append(ax.add_artist(ab))
# #     ax.update_datalim(np.column_stack([x, y]))
# #     ax.autoscale()
# #     return artists
# # 
# # 
# # for depth in range(7,10):
# #     visualizer(depth = depth, DA = False,show_accs = True,plot_domain_disc = False,dim_red = False, plot_entangle = True, tSNE_plot = False, pca_plot = False, isomap_plot=False, LLE_plot = False)
# # 
# # df = pd.DataFrame(d_53s.transpose(), columns=[f'{i}' for i in range(7)])
# # dft = pd.DataFrame(dt_53s.transpose(), columns=[f'{i}' for i in range(7)])
# # # Calculate mean values
# # meanst = dft.mean()
# # means = df.mean()
# # dfd = pd.DataFrame(d_53d.transpose(), columns=[f'{i}' for i in range(7)])
# # dftd = pd.DataFrame(dt_53d.transpose(), columns=[f'{i}' for i in range(7)])
# # # Calculate mean values
# # meanstd = dftd.mean()
# # meansd = dfd.mean()
# # # Create a violin plot with mean points and outliers
# # #ax = sns.violinplot(data=df, inner="points", palette="pastel", showmeans=True, showfliers=True)
# # plt.figure(figsize = (6,4))
# # #plt.boxplot(d_3.transpose(), vert=True, patch_artist=True)
# # plt.plot(range(7), means.values, marker='o', linestyle='--', color='orange', label='Source Mean: blind DA')
# # plt.plot(range(7), meanst.values, marker='o', linestyle='--', color='green', label='Target Mean: blind DA')
# # 
# # plt.plot(range(7), meansd.values, marker='*', linestyle=':', color='orange', label='Source Mean: DANN')
# # plt.plot(range(7), meanstd.values, marker='*', linestyle=':', color='green', label='Target Mean: DANN')
# # # Customize the plot
# # plt.title('MNIST-r to MNIST-c', fontsize=18)
# # plt.xlabel('Representation depth', fontsize=16)
# # plt.ylabel('Class entanglement', fontsize=16)
# # plt.ylim(0,2)
# # plt.xlim(0,6)
# # #plt.xticks(range(1, 12), [f'{i}' for i in range(11)])
# # # Add a legend for mean points and mean line
# # plt.legend( loc="upper right")
# # plt.show()
# # =============================================================================
# 
# =============================================================================
