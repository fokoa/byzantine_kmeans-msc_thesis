#!usr/bin/python3
# -*- coding : utf8 -*-

from mpi4py import MPI;

import numpy as np;

from functions import *;


def methode_1(data, idx_by_cluster):
    """ 
        A method that generates byzantine errors

        Parameters :
        -----------
        data : ndarray of shape (n_samples, n_features)
            Samples to be clustered

        idx_by_cluster : ndarray of shape (n_clusters, length(X_worker))   
            X_worker is the samples assigned to associated cluster
            Row 0 contains the indexes of the samples which in cluster 0
            Row 1 contains the indexes of the samples which in cluster 1
            ...

        Return :
        -------
        centroids : ndarray of shape (n_clusters, n_features)
            Byzantine centroids.         
    """

    n_cluster, n_features = idx_by_cluster.shape[0], data.shape[1]; 
    centroids = np.zeros((n_cluster, n_features));
    
    for cluster in range(0, n_cluster):
        idx = idx_by_cluster[cluster];
        temp_idx = [x for x in idx if x != -1];

        if temp_idx == []:
            pass;
        else:
            idx_selected = np.random.random_integers(0, len(temp_idx)-1);
            centroids[cluster] = sparse_or_not(data[temp_idx[idx_selected]]);

    return centroids;


def selected_13(idx_by_cluster, samp_per_cluster23):
    """ 
        Select one third of the points in each cluster

        Parameters :
        -----------
        idx_by_cluster : ndarray of shape (n_clusters, length(X_worker))   
            X_worker is the samples assigned to associated cluster
            Row 0 contains the indexes of the samples which in cluster 0
            Row 1 contains the indexes of the samples which in cluster 1
            ...

        samp_per_cluster23 : ndarray of shape (n_clusters,)
            Each index contain the number one third  of samples
            that each cluster has

        Return :
        -------
        final_idx : ndarray of shape (n_clusters, length(X_worker))
            Contains the indexes of one third of samples which
            is in each cluster          
    """
    
    final_idx = np.ones(idx_by_cluster.shape);

    for cluster in range(0, idx_by_cluster.shape[0]):
        idx = idx_by_cluster[cluster];
        temp_idx = [x for x in idx if x != -1];

        uni_idx = [];
        repeat = samp_per_cluster23[cluster];
    
        while repeat != 0:
            x = np.random.random_integers(0, len(temp_idx)-1);
            if temp_idx[x] not in uni_idx:
                uni_idx.append(temp_idx[x]);
                repeat = repeat - 1;
        l = []
        for x in idx_by_cluster[cluster]:
            if x in uni_idx:
                l.append(x);
            else: 
                l.append(-1);

        final_idx[cluster] = np.array(l); 
    
    return final_idx;


def methode_2(samp_per_cluster, idx_by_cluster, data, n_clusters):
    """ 
        A method that generates byzantine errors

        Parameters :
        -----------
        samp_per_cluster : ndarray of shape (n_clusters, )
            Contains the number of samples that each cluster has
            Index 0 contains the number of the samples which in cluster 0
            Index 1 contains the number of the samples which in cluster 1
            ...

        idx_by_cluster : ndarray of shape (n_clusters, length(X_worker))   
            X_worker is the samples assigned to associated cluster
            Row 0 contains the indexes of the samples which in cluster 0
            Row 1 contains the indexes of the samples which in cluster 1
            ...

        data : ndarray of shape(n_samples, n_features)
            All samples

        n_cluster : int
            Number of clusters

        Return :
        -------
        centroids, samp_per_clusters23 : tuple of length 2
            First element is an array which contains byzantine centroids
            Second is an array which contains the number of samples that
            each cluster has          
    """

    n_samples, n_features = data.shape;
    samp_per_cluster23 = (2 / 3 * samp_per_cluster).astype(int);
    final_idx_by_cluster = selected_13(idx_by_cluster, samp_per_cluster23);

    # Array which will store the sum of the samples when they are assigned to the cluster
    # (centroid) closest to them. Then we'll just do an arithmetic mean to find the new centroids
    centroids = np.zeros((n_clusters, n_features), dtype=float);
    
    for j in range(0, n_clusters):
        for sample in [int(x) for x in final_idx_by_cluster[j] if x != -1]:
            centroids[j] = centroids[j] + data[sample] 

    return centroids, samp_per_cluster23;


def agregated_centroids(good_centroids, bad_centroids):
    """
        Aggregate all centroids computed by workers
        
        Parameters :
        -----------
        good_centroids : ndarray of shape (n_workers, n_clusters, n_features)
            All centroids computed by the good workers

        bad_centroids : ndarray of shape (n_workers, n_clusters, n_features)
            All centroids computed by the bad workers

        Return :
        -------
        final_centroids : ndarray of shape (n_clusters, n_features)
            Merged centroids.  
    """

    worker_centroids = np.vstack((good_centroids, bad_centroids));
    n_workers, n_clusters, n_features = worker_centroids.shape;
    centroids = np.zeros((n_clusters, n_features));

    for cluster in range(0, n_clusters):
        centroids[cluster] = worker_centroids[:, cluster, :].sum(axis=0);

    return centroids;


def faba(good_centroids, bad_centroids, n_byzan):
    """
        Fix byzantine (bad) centroids according FABA's method
        
        Parameters :
        -----------
        good_centroids : ndarray of shape (n_workers, n_clusters, n_features)
            All centroids computed by the good workers

        bad_centroids : ndarray of shape (n_workers, n_clusters, n_features)
            All centroids computed by the bad workers

        n_byzan : int
            Number of byzantine workers
            
        Return :
        -------
        final_centroids : ndarray of shape (n_clusters, n_features)
            Merged centroids
    """
    
    all_centroids = np.vstack((good_centroids, bad_centroids));
    agregated_centroids = np.zeros((good_centroids.shape[1], good_centroids.shape[2]));

    for n_cluster in range(0, good_centroids.shape[1]):

        centroids = all_centroids[:, n_cluster, :];
        
        k = 1;
        while k < n_byzan:

            # Compute mean all of centroids
            mean_centroids = np.mean(centroids, axis=0);

            # Delete centroid that has the largest difference from all_centroids
            distances = np.zeros((centroids.shape[0], ));
            
            for idx in range(0, centroids.shape[0]):
                norm = np.linalg.norm(mean_centroids - centroids[idx]);
                distances[idx] = norm;

            # The largest
            largest = distances.argmax();
            centroids = np.delete(centroids, largest, axis=0);

            k = k + 1;

        agregated_centroids[n_cluster] = np.mean(centroids, axis=0);

    return agregated_centroids;


class KMeansResilient:
    ''' 
        K-means clustering for grouping data into K groups of similar objects

        Parameters
        ----------
        n_iter : int, default=100
            Number of iterations that will do k-means

        init : 'kmeans++' or random', default='kmeans++'
            Centroids initialization method

        n_clusters : int, default=3
            Number of clusters or centroids where the samples
            will be grouped. It is famous K of K-means.

        n_init : int, default=10
            Number of time the k-means algorithm will be run with
            different centroid seeds. The final results will be
            the best output of n_init consecutive runs in terms of inertia.

        seed : int, default=None
            For the reproducibility

        n_byzantines : int, default=0
            Number of byzantine machines, nodes or workers present
            in the distributed system

        correction : Boolean, default=False.
            Tell us if we must to correct generate byzantine worker

        init_centroids : ndarray of shape (n_init, n_clusters, n_features);
            Contains all the initial centroids used to fit k-means

        Attributes
        ----------
        centroids_ : ndarray of shape (n_clusters, n_features)
            Coordinates of cluster centers.

        labels_ : ndarray of shape (n_samples,)
            Labels of each samples
       
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center
    '''


    def __init__(self, n_clusters=3, init='kmeans++', n_init=10, n_iter=100,
                 seed=None, n_byzantines=0, correction=False): 

        # Check n_clusters 
        if isinstance(n_clusters, int) is False or n_clusters <= 0:
            raise ValueError("'n_clusters' must be an integer and"
                              " strictly greater than 0. You "
                              "gave %s." % str(n_clusters));

        # Check init
        names_init = ['random', 'kmeans++']; 
        if init not in names_init:
            raise ValueError("'init' can only take one of three"
                             " values : 'random' or 'kmeans++'"
                             ". You gave %s." % str(init));

        # Check n_init
        if isinstance(n_init, int) is False or n_init <= 0:
            raise ValueError("'n_init' must be an integer and" 
                             "strictly greater than 0."
                             "You gave %s." % str(n_init));

        # Check n_iter
        if isinstance(n_iter, int) is False or n_iter <= 0:
            raise ValueError("'n_iter' must be an integer and"
                             "be strictly greater than 0."
                             "You gave %s." % str(n_iter)); 
        
        # Check seed
        if seed is not None and (isinstance(seed, int)
                is not True or seed <= 0):
            raise ValueError("'seed' must be an integer and strictly "
                             "greater than 0. You gave %s." % str(seed));

        # n_byzantines
        if isinstance(n_byzantines, int) is False or n_byzantines < 0:
            raise ValueError("'n_byzantines' must be an integer and" 
                             " greater than 0. You got %s"
                             %(str(n_byzantines)));

        # Check correction
        if isinstance(correction, bool) is False:
            raise ValueError("'correction' must be an boolean."
                             " You got %s" %(str(correction)));
        elif correction is True and n_byzantines == 0:
            raise ValueError("If 'correction' is True then 'n_byzantines'"
                             "must be differnet of 0. You got"
                             " %s" %(str(correction)));

        # Initialization
        self.n_clusters = n_clusters;
        self.init = init;
        self.n_init = n_init;
        self.n_iter = n_iter;
        self.seed = seed;
        self.n_byzantines = n_byzantines;
        self.correction = correction;


    def fit(self, X, methode_gen='methode_1'):
        """"
            Fit the clusters

            Parameters :
            -----------
            X : ndarray of shape (n_samples, n_features)
                Samples to be clustered

            methode_gen : str, default="methode_1"
                Byzantine error generation method
                
            Return :
            -------
            self
        """

        # MPI initialization
        comm = MPI.COMM_WORLD;
        P = comm.Get_size(); # Number of workers, including coordinator
        rank = comm.Get_rank();

        n_samples, n_features = X.shape;

        if rank == 0:
            # Store all centroids, labels, 
            # initialization centroids and inertia
            all_centroids = np.zeros((self.n_init, self.n_clusters,
                                      n_features));
            all_labels = np.zeros((self.n_init, n_samples), dtype=int);
            self.init_centroids = np.zeros((self.n_init, self.n_clusters,
                                            n_features));
            all_inertia  = np.zeros((self.n_init, ));

            # Let's choice good and bad workers
            goods, byzantines = choose_byzantines(P, self.n_byzantines,
                                                  self.seed);
            
            # Index of value to distribute
            index_data = get_index(X, P-1, self.seed);
        
        else:
            byzantines = [];
            goods = [];
            index_data = [];

        # Broadcast goods and byzantines
        byzantines = comm.bcast(byzantines, root=0);
        goods = comm.bcast(goods, root=0);

        # Let's distribute data (index) at each machine
        index_data = comm.bcast(index_data, root=0);

        # Let's run k-means a finite number of times in order
        # to select the one with the smallest inertia
        for idx in range(0, self.n_init):
            
            if rank == 0:

                self.inertia_ = 0.0;
                self.labels_ = np.zeros((n_samples, ), dtype=int);

                # Initializing cluster centroids
                if self.init == 'random':
                    self.centroids_ = rand_initialisation(X, self.n_clusters,
                                                          self.seed, idx);
                    self.init_centroids[idx] = self.centroids_.copy();

                elif self.init == 'kmeans++':
                    self.centroids_ = kmeans_plus_plus(X, self.n_clusters,
                                                       self.seed, idx);
                    self.init_centroids[idx] = self.centroids_.copy();

                # Store centroids of the good machines to fix byzantines
                good_centroids = np.zeros((P - 1 - self.n_byzantines,
                                           self.n_clusters, n_features));

                # Store centroids of the bad machines to fix byzantines
                bad_centroids = np.zeros((self.n_byzantines,
                                          self.n_clusters, n_features));
                
            else:
                self.centroids_ = np.zeros((self.n_clusters, n_features));


            for iteration in range(0, self.n_iter):

                inertia = 0.0;

                # Broadcast centroids
                self.centroids_ = comm.bcast(self.centroids_, root=0);

                # Array which will store the sum of the samples when assigned
                # to the cluster (centroid) closest to them. Then we'll just
                # do an  arithmetic mean to find the new centroids
                centroids = np.zeros((self.n_clusters, n_features));
                
                # Array which will contain the number of samples
                # assigned to each cluster
                samp_per_cluster = np.zeros((self.n_clusters, ), dtype=int);

                labels = np.zeros( (n_samples, ), dtype=int)

                if rank != 0:

                    idx_by_cluster = -1 * np.ones((self.n_clusters, 
                                        len(index_data[rank-1])), dtype='int');

                    for sample in index_data[rank-1]:
                        
                        # We will calculate in all self.n_cluster_ 
                        # distances for the sample
                        dist_samp_clusters = np.zeros((self.n_clusters, ));
                    
                        for cluster in range(0, self.n_clusters):
                            norm = np.linalg.norm(sparse_or_not(X[sample]) 
                                                  - self.centroids_[cluster]);
                            dist_samp_clusters[cluster] = np.square(norm);

                        # Find the closest cluster to sample
                        closest_cluster = dist_samp_clusters.argmin();

                        # Join to the sample its cluster
                        labels[sample] = closest_cluster;

                        # Add this sample to the associated cluster
                        centroids[closest_cluster] = centroids[closest_cluster] \
                                                     + sparse_or_not(X[sample]);

                        # Store index of the samples
                        i = 0;
                        fill = False;
                        while fill == False:
                            if idx_by_cluster[closest_cluster][i] == -1:
                                idx_by_cluster[closest_cluster][i] = sample;
                                fill = True;
                            else:
                                i = i + 1;

                        # Let's increment the number of samples 
                        # contained in the cluster
                        samp_per_cluster[closest_cluster] = samp_per_cluster[closest_cluster] + 1;

                        # Add this distance to the inertia
                        inertia = inertia + dist_samp_clusters[closest_cluster];   


                    if self.correction is True:
                        for idx_centroid in range(0, self.n_clusters):
                            if samp_per_cluster[idx_centroid] == 0:
                                centroids[idx_centroid] = centroids[idx_centroid]; 
                            else:
                                centroids[idx_centroid] = (1 / samp_per_cluster[idx_centroid])\
                                                            * centroids[idx_centroid];

                    # Let's change centroids of byzantine machines
                    if rank in byzantines:

                        if methode_gen == 'methode_1':
                            centroids = methode_1(X, idx_by_cluster);

                        elif methode_gen == 'methode_2':
                            centroids, samp_per_cluster = methode_2(samp_per_cluster,
                                                idx_by_cluster, X, self.n_clusters);
                        else:
                            raise ValueError("Unknown generation's methode !"
                                             " There are two methods : "
                                             "'methode_1' and 'methode_2'");


                # Good machines send their centroids to coordinator
                for i, good in enumerate(goods):
                    
                    if rank == good:
                        comm.Send(centroids, dest=0, tag=i);
                    elif rank == 0:
                        comm.Recv(good_centroids[i], source=good, tag=i);           


                # Bad machines send their centroids to coordinator
                for i, bad in enumerate(byzantines): 
                    
                    if rank == bad:
                        comm.Send(centroids, dest=0, tag=i);
                    elif rank == 0:
                        comm.Recv(bad_centroids[i], source=bad, tag=i);


                # Join all labels
                self.labels_ = comm.reduce(labels, MPI.SUM, 0);

                # Let's sum arrays which will contain the numbers of
                # samples assigned to each cluster
                samp_per_cluster = comm.reduce(samp_per_cluster, MPI.SUM, 0);

                if rank == 0:
                    
                    if self.correction is True:
                        centroids = faba(good_centroids, bad_centroids,
                                         self.n_byzantines);

                    else:
                        centroids = agregated_centroids(good_centroids,
                                                        bad_centroids);
                        
                        # Let's calculate the new centroids
                        for cluster in range(0, self.n_clusters):
                            # 1 when the cluster does not contain any
                            # example apart from its centroid
                            samp_per_cluster[cluster] = max([samp_per_cluster[cluster], 1]);
                            
                            # New centroids
                            self.centroids_[cluster] = centroids[cluster] / samp_per_cluster[cluster]; 
                        

            self.inertia_ = comm.reduce(inertia, MPI.SUM, 0);

            if rank == 0:
                all_centroids[idx] = self.centroids_;
                all_labels[idx] = self.labels_;
                all_inertia[idx]  = self.inertia_;


        if rank == 0:
            # Let's select the best k initial centroids,
            # k centroids, labels and inertia according
            # the ones that have the smallest inertia
            self.inertia_ = all_inertia[all_inertia.argmin()];
            self.centroids_ = all_centroids[all_inertia.argmin()]; 
            self.labels_ = all_labels[all_inertia.argmin()];
            

        return self;


    def predict(self, X):
        """
            Predict clusters

            Parameters :
            -----------
            X : ndarray of shape (n_samples, n_features)
                Samples to be clustered
    
            Return :
            -------
            predictions : ndarray of shape (n_samples,)
                Labels of X
        """

        n_samples, n_features = X.shape;
        predictions = np.zeros((n_samples, ), dtype=int)

        for sample in range(0, n_samples):
                    
            # Distance from a sample to all centroids
            dist_samp_clusters = np.zeros((self.n_clusters, ));
            
            for cluster in range(0, self.n_clusters):
                norm = np.linalg.norm(sparse_or_not(X[sample]) - 
                                      self.centroids_[cluster]);
                dist_samp_clusters[cluster] = np.square(norm);

            # Find the closest cluster  to sample
            closest_cluster = dist_samp_clusters.argmin();

            # Join sample and its cluster
            predictions[sample] = closest_cluster;
        
        return predictions


    def misclassified(self, X, label_km, label_byzan):
        """
            Count the number of misclassified points
            compared to that of kmeans

            Parameters :
            -----------
            label_km : ndarray of shape (n_samples)
               Label computed by K-means

            label_byzan : ndarray of shape (n_samples)
                Label computed by byzantine K-means

            Return :
            -------
            n_misclass : int
                Total number of misclassified samples
        """

        misclass = [];

        for cluster in range(0, self.n_clusters):
            count_km = count_value(cluster, label_km);
            count_byzan = count_value(cluster, label_byzan);

            misclass.append(abs(count_km - count_byzan));
        
        return max(misclass);


