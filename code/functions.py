#!usr/bin/python3
# -*- coding : utf8 -*-

import random;
import numpy as np;

import scipy.sparse as sp;
from sklearn.model_selection import KFold;
from sklearn.preprocessing import StandardScaler, LabelEncoder;



def sparse_or_not(x, is_matrix=False):
    """ 
        Convert sparse vector or sparse matrix to ndarray
        If sparse, convert it to ndarray and return it
        Else, return it

        Parameters :
        -----------
        x : sparse or ndarray
            Sparse vector or sparse matrix or ndarray

        is_matrix : boolean, default=False   
            Tell us if x is an vector or matrix

        Return :
        -------
        y : ndarray         
    """

    if sp.issparse(x):
        if is_matrix:
            return x.toarray();
        else:
            return x.toarray().reshape((-1, ));
    else:
        return x;


def rand_initialisation(X, n_clusters, seed, cste):
    """ 
        Initialize centroids from X randomly 

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to be clustered

        n_clusters : int   
            Number of clusters / Number of centroids

        seed : int
            For reproductibility

        cste : int
            To add to seed for reproductibility

        Return :
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Initial centroids
    """

    index = [];
    repeat = n_clusters;

    # Take one index
    if seed is None:
        idx = np.random.RandomState().randint(X.shape[0]);
    else:
        idx = np.random.RandomState(seed+cste).randint(X.shape[0]);
    
    while repeat != 0:

        # Let's check that we haven't taken this index yet
        if idx not in index:
            index.append(idx);
            repeat = repeat - 1;

        if seed is not None:
            idx = np.random.RandomState(seed+cste+repeat).randint(X.shape[0]);

    return sparse_or_not(X[index], is_matrix=True);


def kmeans_plus_plus(X, n_clusters, seed, cste):
    """ 
        Initialize centroids from X according heuristic kmeans++ 

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to be clustered

        n_clusters : int   
            Number of clusters / Number of centroids

        seed : int
            For reproductibility

        cste : int
            To add to seed for reproductibility

        Return :
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Initial centroids
    """

    n_samples, n_features = X.shape;
    centroids = [];

    # First centroid is randomly selected from the data points X
    if seed is None:
        centroids.append( sparse_or_not(X[np.random.RandomState()
                                            .randint(X.shape[0])]) );
    else:
        centroids.append( sparse_or_not(X[np.random.RandomState(seed+cste)
                                            .randint(X.shape[0])]) );

    # Let's select remaining "n_clusters - 1" centroids
    for cluster_idx in range(1, n_clusters):

        # Array that will store distances of data points from nearest centroid
        distances = np.zeros((n_samples, ));

        for sample_idx in range(n_samples):
            minimum = np.inf;
              
            # Let's compute distance of 'point' from each of the previously
            # selected centroid and store the minimum distance
            for j in range(0, len(centroids)):

                dist = np.square( np.linalg.norm(sparse_or_not(X[sample_idx]) - centroids[j]) );
                minimum = min(minimum, dist);
            
            distances[sample_idx] = minimum;
        
        centroids.append(sparse_or_not(X[np.argmax(distances)]));
    
    return np.array(centroids);


def choose_byzantines(P, n_byzantines, seed):
    """ 
        Generate machines that will become good and byzantines

        Parameters :
        -----------
        P : int
            Number of nodes (machines)
            0 is the coordinator (server) ID
            {1, 2,..., P-1} workers ID

        n_byzantine : int   
            Number of byzantines nodes

        seed : int
            For reproductibility

        Return :
        -------
        goods, byzantines : tuple of length 2
            goods is the list of good workers
            byzantines is the list of bad workers
    """

    byzantines = [];
    repeat = n_byzantines;
    cste = 1;
    
    while repeat != 0:
        
        # Take one index
        if seed is None:
            x = np.random.RandomState().randint(1, P);
        else:
            x = np.random.RandomState(seed+cste).randint(1, P);
            cste = cste + 2;
        
        if x not in byzantines:
            byzantines.append(x);
            repeat = repeat -1;
    
    goods = [x for x in range(1, P) if x not in byzantines];

    return goods, byzantines;


def get_index(X, n_worker, seed_data):
    """"
        Distribute data to workers

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to be clustered 

        n_worker : int   
            Number of workers

        seed_data : int
            For reproductibility

        Return :
        -------
        index : list of length 'n_worker'
            Element 1 contain an index list of samples
            assigned to worker 1
            Element 2 contain an index list of samples
            assigned to worker 2. So on
    """

    index = [];
    if seed_data is None:
        seed_data = 42;

    kfold = KFold(n_splits=n_worker, shuffle=True, random_state=seed_data);
    
    for _, idx in kfold.split(X):
        index.append(idx);

    return index;


def count_value(value, values):
    """
        Count the number of appearances

        Parameters :
        -----------
        value : int
            Cluster label 

        values : ndarray of shape (n_samples, )
            Sample labels

        Return :
        -------
        count : int
            Number of times 'value' appear in 'values'
    """

    count = 0;
    for val in values:
        if val == value:
            count = count + 1;

    return count;


