#!usr/bin/python3
# -*- coding : utf8 -*-


import sys;
import getopt;
from mpi4py import MPI;

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

from sklearn import decomposition;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;

from kmeans_resilient import KMeansResilient as KMR;
# from functions import *;


# MPI initialization
comm = MPI.COMM_WORLD;
P = comm.Get_size();
rank = comm.Get_rank();


def check_option(opt_arg):
    """ 
        Check the arguments passed as parameters by the
        command prompt 

        Parameters :
        -----------
        opt_arg : str
            Arguments and options passed by the command prompt

        Return :
        -------
        opts : list
            Argument list 
        args : list
            Option list
    """

    try:
        opts, args = opt_arg;

    except getopt.GetoptError as err:
        print(err);
        print("Use :\t", sys.argv[0], "-b 5 \n\t",
            "or:", sys.argv[0], "--byzantine 5");
        sys.exit(-1);

    for opt, val in opts:

        if opt in ("-b", "--byzantine"):
            if val.isdigit() == False:
                raise ValueError("Enter an integer as number"
                                 "of byzantine machines.");

        elif opt in ("-h", "--help"):
            print("Use:", sys.argv[0], "-b 5\n",
            "or:", sys.argv[0], "--byzantine 5");
            sys.exit(-1);
        
        else:
            print("unhandled options");
            sys.exit(-1);

    return opts, args;



def check_Nbyzan(opts, P):
    """ 
        Check and get the number of Byzantine machines that
        we are going to simulate 

        Parameters :
        -----------
        opts : str
            Options passed by the command prompt

        P : int
            Total number of machines (nodes or workers).
            1 coodinator ans the ramaining are workers

        Return :
        -------
        n_byzantines : int (entire natural)
            Number of byzantine machines that we
            are going to simulate
    """

    if len(opts) == 0:
        n_byzantines = 0;
        
    n_byzantines = int(opts[0][1]);
    
    if n_byzantines < 0 or n_byzantines > P - 1:
        raise ValueError("Number of byzantine must be an integer "
                         "< number of workers or >= 0");
           
    return n_byzantines;



def sort_centroides(centroids):
    """ 
        Sort centroids according to their norms

        Parameters :
        -----------
        centroids : ndarray of shape (k, n_features)
            All centroids of clusters where k
            is number of clusters

        Return :
        -------
        tmp : ndarray of shape (k, n_features)
            Sorted centroids
    """

    tmp = np.zeros((centroids.shape));
    normes = {};

    for centroid in range(0, centroids.shape[0]):
        norm = np.linalg.norm(centroids[centroid]);
        normes[norm] = centroid;

    i=0;
    for norm in sorted(normes):
        tmp[i] = centroids[normes[norm]];
        i = i + 1;

    return tmp;
    


def comparaison(X, label_km, label_by, label_co):
    """ 
        Plot all the formed clusters 

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to be clustered
        
        label_km : list of length 2
            The first is labels obtained with K-means
            The second is number of clusters

        label_by : list of length 2
            The first is labels obtained with byzantin K-means
            The second is number of byzantines

        label_co : ndarray of shape (n_samples, )
            Label obtained by correcting byzantines 
            in byzantin K-means
    """

    plt.figure(1, figsize=(10,4), facecolor="w");

    plt.subplot(131);
    plt.scatter(X[:, 0], X[:, 1], c=label_km[0]);
    plt.title('%d-means'%(label_km[1]));

    plt.subplot(132);
    plt.scatter(X[:, 0], X[:, 1], c=label_by[0]);
    plt.title('%d Byzantines' % (label_by[1]) );

    plt.subplot(133);
    plt.scatter(X[:, 0], X[:, 1], c=label_co);
    plt.title('Correction');

    # plt.savefig('km_by_co_11.pdf', format='pdf');
    plt.show();



def main():

    # Check options and number of byzantines
    opts, arg = check_option(getopt.getopt(sys.argv[1:], "hb:",
                             ["help", "byzantine="]));
    n_byzantines = check_Nbyzan(opts, P);

    # Load dataset
    iris = pd.read_csv('data/iris.data', delimiter=',',
                       names=['a', 'b', 'c', 'd', 'class']);

    # Parameters
    datasets = [ (iris, {'n_cluster':3, 'name':'iris'})
               ];

    for data in datasets:
    
        X = data[0][['a', 'b', 'c', 'd']].values;
        km = KMR(n_clusters=data[1]['n_cluster'], n_init=1,
                n_iter=50, seed=7);
        by = KMR(n_clusters=data[1]['n_cluster'], n_init=1,
                n_iter=50, seed=7,n_byzantines=n_byzantines);
        co = KMR(n_clusters=data[1]['n_cluster'], n_init=1,
                n_iter=50, seed=7, n_byzantines=n_byzantines,
                correction=True);

        
        # Fit
        km.fit(X);
        by.fit(X);
        co.fit(X);

        # Sort centroides
        km.centroids_ = sort_centroides(km.centroids_);
        by.centroids_ = sort_centroides(by.centroids_);
        co.centroids_ = sort_centroides(co.centroids_);
        
        # Plot
        if rank == 0:

            print('##########################################');
            print('%s dataset' % (data[1]['name']));
            print('##########################################');

            print('\nKmeans centroids:\n' , km.centroids_);
            print('Byzantine centroids:\n', by.centroids_);
            print('Correct centroids:\n', co.centroids_);

            print('\nKmeans inertia:\n', km.inertia_);
            print('\nByzantine inertia:\n', by.inertia_);
            print('\nCorrection inertia:\n', co.inertia_);

            mis_1 = by.misclassified(X, km.labels_, by.labels_);
            mis_2 = co.misclassified(X, km.labels_, co.labels_);

            print('\nByzantine has %d data point misclassified.' % (mis_1));
            print('\nCorrection has %d data point misclassified.' % (mis_2));

            if data[1]['name'] == 'iris':
                pca = decomposition.PCA(n_components=2);
                X = pca.fit_transform(data[0][['a', 'b', 'c', 'd']].values);
            
            comparaison(X, [km.labels_, km.n_clusters],
                        [by.labels_, by.n_byzantines], co.labels_);



if __name__ == "__main__":
    main();

