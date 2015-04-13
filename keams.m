%kmeans with different distance and start 
%calculate accuracy
% applies the k-means clustering algorithm to the test set
[idxs,centroids] = kmeans(tstmtx',5,'Distance','sqEuclidean','Start','sample');

