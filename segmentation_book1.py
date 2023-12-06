# -*- coding: utf-8 -*-
"""Segmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Axzu443HUAGK6qVj6cuJbo6_0y45SqWB

# Starting

Run the following code to import the modules and download all the files you'll need.. After your finish the assignment, remember to run all cells and save the note book to your local machine as a .ipynb file for Canvas submission.
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy
# from google.colab.patches import cv2_imshow
from skimage import segmentation
from skimage import io, color
import skimage

def cluster_centers(superpixel_map):
  """
  This function takes a superpixel map and returns a list with the
  (row,col) positions of the cluster centers for that map
  """

  unique_labels = np.unique(superpixel_map)
  cluster_center_list = []

  for label_id, superpixel_label in enumerate(unique_labels):

      # Compute the coordinates where we have the superpixel_map = current label

      cluster_indices = np.where(superpixel_map == superpixel_label)

      # Compute the centroid of the coordinates to find the cluster centers

      centers = np.round(np.mean(cluster_indices, axis = 1)).astype('int')
      cluster_center_list.append(centers)

  return cluster_center_list

"""#Implement color histogram

You will implement the function color_histogram next. This function takes an image, a binary mask and the number of bins that you want to use for every channel. The function will compute the histogram over the 1 values in the mask only.
"""

def color_histogram(img, mask, num_bins):
  """For each channel in the image, compute a color histogram with the number of bins
  given by num_bins of the pixels in
  image where the mask is true. Then, concatenate the vectors together into one column vector (first
  channel at top).

  Mask is a matrix of booleans the same size as image.

  You MUST normalize the histogram of EACH CHANNEL so that it sums to 1.
  You CAN use the numpy.histogram function.
  You MAY loop over the channels.
  The output should be a 3*num_bins vector because we have a color image and
  you have a separate histogram per color channel.

  Hint: np.histogram(img[:,:,channel][mask], num_bins)"""

  rows, cols, channels = img.shape
  histogram = []

  for ch in range(channels):
    hist = np.histogram(img[:,:,ch][mask], num_bins)[0]
    histogram.append(hist/np.sum(hist))

  return np.array(histogram).reshape(3*num_bins)

"""#Implement adjacency matrix

You need to implement the adjacency matrix function that takes a superpixel map as an input and outputs a binary adjacency matrix.
"""

def adjacencyMatrix(superpixel_map):
  """Implement the code to compute the adjacency matrix for the superpixel map
  The input is a superpixel map and the output is a binary adjacency matrix NxN
  (N being the number of superpixels in svMap).  Bmap has a 1 in cell i,j if
  superpixel i and j are neighbors. Otherwise, it has a 0.  Superpixels are neighbors
  if any of their pixels are neighbors."""

  segmentList = np.unique(superpixel_map)
  segmentNum = len(segmentList)
  adjMatrix = np.zeros((segmentNum, segmentNum))

  # for i in range(superpixel_map.shape[0]-1):
  #   for j in range(superpixel_map.shape[1]-1):
  #     if(superpixel_map[i,j] != superpixel_map[i+1,j]): # if the super pixels are adjacent
  #       adjMatrix[superpixel_map[i,j],superpixel_map[i+1,j]] = 1
  #       adjMatrix[superpixel_map[i+1,j],superpixel_map[i,j]] = 1
  #     if(superpixel_map[i,j] != superpixel_map[i,j+1]):
  #       adjMatrix[superpixel_map[i,j],superpixel_map[i,j+1]] = 1
  #       adjMatrix[superpixel_map[i,j+1],superpixel_map[i,j]] = 1

  # Cody
  adjMatrix[superpixel_map[:, :-1], superpixel_map[:, 1:]] = superpixel_map[:, :-1] != superpixel_map[:, 1:]
  adjMatrix[superpixel_map[:, 1:], superpixel_map[:, :-1]] = superpixel_map[:, 1:] != superpixel_map[:, :-1]

  adjMatrix[superpixel_map[:-1, :], superpixel_map[1:, :]] = superpixel_map[:-1, :] != superpixel_map[1:, :]
  adjMatrix[superpixel_map[1:, :], superpixel_map[:-1, :]] = superpixel_map[1:, :] != superpixel_map[:-1, :]

  return adjMatrix

"""#Implement your graph-cut algorithm

It is time to build your foreground-background segmentation algorithm. For this we provide you with the implementation of the Ford-fulkerson algorithm which you will need to determine where your graph should be cut. You can learn more about the algorithm here: https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/. Please note that our version of the Ford Fulkerson algorithm doesn't return the max flow as a scalar. It returns the current flow through each edge of the graph when we reach the point of maximum flow.

We also implemented the reduce function which takes an image, its corresponding superpixel map, and a number of bins as input. The output is a list of feature vectors. Each feature vector is the resulting histogram from applying the color_histogram function you implemented to every segment on the superpixel map.

You will implement the graph_cut function

"""

# Python program for implementation of Ford Fulkerson algorithm
# The author of this code is Neelam Yadav

from collections import defaultdict

#This class represents a directed graph using adjacency matrix representation
class Graph:

    def __init__(self,graph):
        self.graph = graph # residual graph
        self. ROW = len(graph)
        # self.COL = len(gr[0])


    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''
    def BFS(self,s, t, parent):

        # Mark all the vertices as not visited
        visited =[False]*(self.ROW)

        # Create a queue for BFS
        queue=[]

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

         # Standard BFS Loop
        while queue:

            #Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0 :
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False


    # Returns tne current flow from s to t in the given graph
    def FordFulkerson(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1]*(self.ROW)

        max_flow = 0 # There is no flow initially
        current_flow = np.zeros_like(self.graph)

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent) :
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while(s !=  source):
                path_flow = min (path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow +=  path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while(v !=  source):
                u = parent[v]
                self.graph[u][v] -= path_flow

                #if u != source:
                self.graph[v][u] += path_flow

                current_flow[u][v] += path_flow
                #if u != source:
                current_flow[v][u] -= path_flow
                v = parent[v]

        return current_flow
    
def apply_supermap(img, superpixel_map):
  """ This function returns an image where we assign the color of the cluster centers
  to every pixel of their corresponding segmentation groups."""
  centers = cluster_centers(superpixel_map)
  out = np.zeros_like(img)
  for i,(row, col) in enumerate(centers):
    out[superpixel_map == i] = img[row, col]
  return out

def reduce(img, superpixel_map, num_bins=10):
  """This function takes as input an image, its corresponding superpixel map, and a
  number of bins as input. The output is a list of feature vectors.
  Each feature vector is the resulting histogram from applying the color_histogram
  function you implemented to every segment on the superpixel map."""

  num_segments = len(np.unique(superpixel_map))
  feature_vectors = []

  for i in range(num_segments):
      mask = superpixel_map == i
      feature_vectors.append(color_histogram(img, mask, num_bins))
  return(feature_vectors)

  # feature_vectors[range(num_segments)] = color_histogram(img, superpixel_map == range(num_segments) , num_bins)
  # # return(feature_vectors)
  # num_segments = len(np.unique(superpixel_map))
  # masks = [superpixel_map == i for i in range(num_segments)]
  # feature_vectors = [color_histogram(img, mask, num_bins) for mask in masks]
  # return feature_vectors


import math

def graph_cut(superpixel_map, features, centers, keyindex):
  """Function to take a superpixel set and a keyindex and convert to a
  foreground/background segmentation.

  keyindex is the index to the superpixel segment we wish to use as foreground and
  find its relevant neighbors.

  centers is a list of tuples (row, col) with the positions of the cluster centers
  of the superpixel_map

  features is a list of histograms (obtained from the reduce function) for every superpixel
  segment in an image.

  """

  #Compute basic adjacency information of superpixels
  #Note that adjacencyMatrix is code you need to implement

  # ===============================================
  # TODO: this should be one line of code

  adjMatrix = adjacencyMatrix(superpixel_map)

  # ===============================================


  # normalization for distance calculation based on the image size
  # for points (x1,y1) and (x2,y2), distance is
  # exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
  dnorm = 2*(superpixel_map.shape[0]/2 *superpixel_map.shape[1] /2)**2
  k = len(features) #number of superpixels in image

  #Generate capacity matrix
  capacity = np.zeros((k+2,k+2))
  source = k
  sink = k+1

  # This is a single planar graph with an extra source and sink
  #  Capacity of a present edge in the graph is to be defined as the product of
  #  1:  the histogram similarity between the two color histogram feature vectors.
  #  The similarity between histograms should be computed as the intersections between
  #  the histograms. i.e: sum(min(histogram 1, histogram 2))
  #  2:  the spatial proximity between the two superpixels connected by the edge.
  #      use exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
  #
  #  Source gets connected to every node except sink
  #  Capacity is with respect to the keyindex superpixel
  #  Sink gets connected to every node except source and its capacity is opposite
  # The weight between a pixel and the sink is going to be the max of all the weights between
  # the source and the image pixels minus the weight between that specific pixel and the source.
  # Other superpixels get connected to each other based on computed adjacency
  # matrix: the capacity is defined as above, EXCEPT THAT YOU ALSO NEED TO MULTIPLY BY A SCALAR 0.25 for
  # adjacent superpixels.


  key_features = features[keyindex] # color histogram representation of superpixel # keyindex
  key_x = centers[keyindex][1] # row of cluster center for superpixel # keyindex
  key_y =  centers[keyindex][0] # col of cluster center for superpixel # keyindex

  # ===============================================
  # TODO: Generate the capacity matrix using the description above. Replace pass with your code

  #Cody
  # for i in range(k):
  #   capacity[source,i] = np.sum(np.minimum(features[i],key_features)) * math.exp(-((centers[i]-centers[keyindex]).sum()**2)/dnorm)
  #   for j in range(k):
  #       capacity[i,j] = np.sum(np.minimum(features[i],features[j])) * math.exp(-((centers[i]-centers[j]).sum()**2)/dnorm) * (adjMatrix[i,j]*0.25)

  # maximum =  max(capacity[source,:])
  # for i in range(k):
  #   capacity[i,sink] = maximum - capacity[source,i]

  # Assuming features, key_features, centers, adjMatrix, and capacity are NumPy arrays
  # Compute capacity[source, i]
  capacity[source, :-2] = np.sum(np.minimum(features, key_features), axis=1) * np.exp(-np.sum((centers[:-2] - centers[keyindex])**2) / dnorm)

  # Compute capacity[i, j]
  # capacity[:-2, :-2] = np.sum(np.minimum(features[:-2, np.newaxis, :-2], features), axis=2) * np.exp(-np.sum((centers[:-2, np.newaxis, :-2] - centers) ** 2, axis=2) / dnorm) * (adjMatrix * 0.25)
  for i in range(k):
    for j in range(k):
       capacity[i,j] = np.sum(np.minimum(features[i],features[j])) * math.exp(-((centers[i]-centers[j]).sum()**2)/dnorm) * (adjMatrix[i,j]*0.25)

  # Find maximum
  maximum = np.max(capacity[source, :])

  # Compute capacity[i, sink]
  capacity[:-2, sink] = maximum - capacity[source, :-2]


  # ===============================================
  # capacity = (100*capacity).astype('int')
  #capacity = np.round(capacity, 6)
  #capacity = np.transpose(capacity)
  capacity = (1e6*capacity).astype('int') ## Converting to integer.

  # Obtaining the current flow of the graph when the flow is max
  g = Graph(capacity.copy())
  current_flow = g.FordFulkerson(source, sink)

  # Extract the two-class segmentation.
  # the cut will separate all nodes into those connected to the
  # source and those connected to the sink.
  # The current_flow matrix contains the necessary information about
  # the max-flow through the graph.

  segment_map = np.zeros_like(superpixel_map)
  rem_capacity = capacity - current_flow

  # ===============================================
  # TODO: Do the segmentation and fill segmentation map with 1s where the foreground is.
  # Replace pass with your code

  #print(rem_capacity)
  #import pdb; pdb.set_trace()
  rem_g = Graph(rem_capacity.copy())
  for i in range(rem_capacity.shape[1]-1):
    parent = [-1]*(rem_g.ROW)
    if(rem_g.BFS(source, i, parent)):
      #if (parent[i] != source) or (parent[i] != -1):
        #print([i, parent[i]])

      # for s in range(superpixel_map.shape[0]):
      #   for t in range(superpixel_map.shape[1]):
      #     if(superpixel_map[s,t]==i):     # pixel matches node connected to source
      #       segment_map[s,t] = 1

      #Cody
      indices = np.where(superpixel_map==i)
      segment_map[indices] = 1


  # ===============================================
  return capacity, segment_map


def segment(img, x, y):
  img = img[:,:,:3]
  super_img = segmentation.slic(img, n_segments=30, compactness=20) - 1
  super_pix = super_img[x, y]
  super_img[super_img != super_pix] = 0
  super_img[super_img == super_pix] = 255

  # return super_img
  
  # noise removal
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(super_img.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations = 2)
  # sure background area
  sure_bg = cv2.dilate(opening, kernel, iterations=3)
  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)

  # Marker labelling
  ret, markers = cv2.connectedComponents(sure_fg)
  # Add one to all labels so that sure background is not 0, but 1
  markers = markers+1
  # Now, mark the region of unknown with zero
  markers[unknown==255] = 0
  markers = cv2.watershed(img,markers)
  out_put = img.copy()
  #out_put[markers == -1] = [255,0,0]
  #out_put[markers == 1] = [0,255,0]
  #out_put[markers == 2] = [0,0,255]
  
  #out_put[markers != markers[x,y]] = [0,0,0]
  out_put = skimage.color.label2rgb(markers, out_put)
  print(np.unique(markers))

  return out_put

"""
def segment(img, x, y):
  img = img[:,:,:3]
  super_img = segmentation.slic(img, n_segments=50, compactness=20) - 1
  # super_img = segmentation.quickshift(img, ratio=.8, kernel_size=20, max_dist=10)
  super_pix = super_img[x, y]

  img_features = reduce(img, super_img)
  img_centers = cluster_centers(super_img)
  img_capacity, img_segment_map = graph_cut(super_img, img_features, img_centers, super_pix)

  return img_segment_map
"""

#-------------------
# img_in = cv2.imread("C:\\Users\\codyd\\EECS-504\\504 Project\\504 Project\\download.jpg")
# img = segment(img_in, 100, 138)
# # print(img)
# # out = img_in.copy()
# # out[:,:,0] = img*255
# # out[:,:,1] = img*255
# # out[:,:,2] = img*255
# cv2.imshow("output", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
