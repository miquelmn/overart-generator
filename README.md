OverArt Generator
==============================

Algorithm for the generation of the artificial dataset OverArt

OverArt dataset
------------

We generate the OverArt dataset in order to obtain a ground truth of the concave points of 
overlapped objects. Each image of the dataset contains a cluster with three overlapped ellipses. 
The use of three ellipses is a good trade-off between complexity and reality. The proposed method to
detect concave points is only necessary when there are at least two overlapped objects, combination 
that is the most commonly found in the different clusters.

The ellipses of each image are defined by three parameters: the rotation, the feret diameter size 
and its center. The values of this parameters are generated randomly by the set of constraints 
described in Table I. This constrains are completed with the next restriction: none of the three 
cells must be outside the cluster. The first ellipse is located in the center of the image. The
positions of the other two ellipses are related to the first one. The location of the second ellipse 
is randomly selected inside the area defined by the minimum and maximum distance to the center of 
the first ellipse. Finally, the third one is randomly placed inside the area defined by the minimum 
and maximum distance to the center of the first and second ellipses.