# Intelligent Data and Probabilistic Inference

## The Naive Bayesian Network
In the first coursework we will look at calculating joint and conditional probability tables (link matrices) and
making inferences with a naive Bayesian network.
Take a look at the first four functions in IDAPICourseworkLibrary.py to make sure you understand what is
going on. You may find the python syntax self explanatory, but if not you can find several good online tutorials
through google. The first function reads a data file in the above format, extracting the data so that it can be
used. It is extensively commented to help you understand what is going on. The second writes a data array to
a file, appending it to what is there already. The data is written out in a fixed floating point format which is
suitable for printing probabilities. The third appends a list (one dimensional array or vector) to a text file and
the fourth appends a string to a text file. These allow you to save your results in a text file and add comments
to them
### Task 1.1
Complete the definition of function Prior. It should calculate the prior distribution over the states of the variable
passed as the parameter ’root’ in the data array. The first line simply sets up an array for the result with zero
entries.
### Task 1.2
Complete the definition of CPT which calculates a conditional probability table (or link matrix) between two
variables indicated by parameters varP (the parent) and varC (the child). In the skeleton the conditional probability
table is simply initialised to zeros.
### Task 1.3
Complete the definition of function JPT which calculates the joint probability table of any two variables. The
joint probability is simply the frequency of a state pair occurring in the data. 
the states of B are columns.
### Task 1.4
Complete the function JPT2CPT which calculates a conditional probability table from a joint probability table.
This is purely a matter of normalising each column to sum to 1. In effect we use the equation P(AjB) =
P(A&B)=P(B).
### Task 1.5
Finally complete the function Query which calculates the probability distribution over the root node of a naive
Bayesian network. To represent a naive network in Python we will use a list containing an entry for each node
(in numeric order) giving the associated probability table: [prior, cpt1, cpt2, cpt3, cpt4, cpt5]. You can calculate
the prior of the root and the conditional probability tables between each child variable and the root using your
solutions to Tasks 1 and 2. A query is a list of the instantiated states of the child nodes, for example [1,0,3,2,0].
The returned value is a list (or vector) giving the posterior probability distribution over the states of the root
node, for example [0.1,0.3,0.4,0.2].
### Results File
You should finish by writing a main program part at the end of the skeleton file. Some example code is there to
help you do this. You should use the Neurones.txt data set, and create a results file containing:
1. A title giving your group members
2. The prior probability distribution of node 0 in the data set
3. The conditional probability matrix P(2|0) calculated from the data.
4. The joint probability matrix P(2&0) calculated from the data.
5. The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0).
6. The results of queries [4,0,0,0,5] and [6, 5, 2, 5, 5] on the naive network

## The Maximally Weighted Spanning Tree
This coursework is concerned with the spanning tree algorithm for finding a singly connected Bayesian network
from a data set. The material you need will be completely covered by Lecture 6 of the course. You can use the
same skeleton program that was set up for Coursework 1, continuing to fill in code where indicated. You will
need to use the function that you wrote to compute the joint probability distribution of a pair of variables. 
### Task 2.1
Complete the function “MutualInformation” which calculates the mutual information (or Kullback Leibler
divergence) of two variables from their joint probability table. The process involves marginalising the joint
probability table and then applying the formula as described in Lecture 6.
### Task 2.2
Complete the function “DependencyMatrix” which uses mutual information as a measure and creates a symmetric
matrix showing the pairwise dependencies between the variables in a data set.
### Task 2.3
Complete the function “DependencyList” which turns the dependency matrix into a list or arcs ordered by their
dependency. Your list items should be triplets: [dependency, node1, node2]. Using your dependency list draw
by hand the network for the HepatitisC data set. Make an image file of your network (either by scanning your
hand drawing or using a drawing package (eg powerpoint or open office)).
### Task 2.4
Starting with a dependency list find the maximally weighted spanning tree automatically, and append it as a list
to your results file. Don’t wory about finding the causal directions.
### Results File
You should finish by writing a main program part at the end of the skeleton file which will add the following
items to your results file:
1. A title giving your group members
2. The dependency matrix for the HepatitisC data set
3. The dependency list for the HepatitisC data set.
4. Ths spanning tree found for the HepatitisC data set (if you attempted task 2.4).
Assemble the results listed above with the network picture into a .pdf file (You can do this with word, open
office or latex). 

## The Minimum Description Length Metric
This coursework is concerned with the minimum description length measure of a Bayesian network and an
associated data set. T
### Task 3.1
Complete the function CPT 2 which computes a conditional probability table (or link matrix) for variables with
two parents.
### Task 3.2
Write a network definition for the HepatitisC data set.
### Task 3.3
Complete the definition of function MDLSize which calculates the size of a network, given the definition of the
network in the two list form described above.
### Task 3.4
Complete the definition of the function JointProbability which calculates the joint probability of a single point,
supplied as a list eg [1,0,3,2,1,0], for a given Bayesian network.
### Task 3.5
Complete the definition of function MDLAccuracy which calculates log likelihood of the network given the
data as described in the lectures.
### Task 3.6
Write a function to find the best scoring network formed by deleting one arc from the spanning tree.
### Results File
You should finish by replacing the main program part at the end of the skeleton file with one which will add the
following items to your results file:
1. A title giving your group members
2. The MDLSize of the your network for Hepatitis C data set
3. The MDLAccuracy of the your network for Hepatitis C data set
4. The MDLScore of the your network for Hepatitis C data set
5. The score of your best network with one arc removed

## Principal Component Analysis
This coursework is concerned with covariance estimation and finding principal components. It uses numpy’s
linear algebra module (linalg) and its image handling capabilities. There is no dependence on courseworks
1-3. The data to be used is the HepatitisC data set, the set of six face images a.pgm .. f.pgm, and the file
PrincipalComponents.txt. The file IDAPICourseworkLibrary contains three functions which will read images
and turn them into data sets and vice versa.
## Task 4.1
Complete the function Mean which calculates the mean vector of a data set represented (as usual) by a matrix
in which the rows are data points and the columns are variables.
## Task 4.2
Complete the function Covariance which calculates the covariance matrix of a data set represented as above.
## Task 4.3
For this part you will need to use the file named PrincipalComponents.txt which is on the course web page.
It contains a set of ten principal components (the last having a zero eigenvalue) calculated from ten different
images of the subject in image c.pmg. You can read it with the function ReadEigenfaceBasis() which is in
the file IDAPICourseworkLibrary.py. It returns an array where each row is an eigenface. You will also need
the mean face for this basis which is in the file MeanImage.jpg. You can read this in with the library function
ReadOneImage(filename).
Complete the function CreateEigenfaceFiles to create image files of the principal components (eigenfaces)
in the format returned by ReadEigenfaceBasis(). You can use the IDAPICourseworkLibrary file SaveEigenface
to do the image handling. Don’t use the file format .pgm (eg instead choose .jpg or .png) when creating these
image files. This is to avoid problems later. You can create a series of file names for the different eigenfaces
within a loop using python’s powerful string concatination feature, for example: filename = “PrincipalComponent”+ str(j) + “.jpg”
## Task 4.4
Complete the function ProjectFace, which reads one image file (use c.pmg for your example) and projects it
onto the principal component basis.
## Task 4.5
Complete the function CreatePartialReconstructions which generates and saves image files of the reconstruction
of an image from its component magnitudes. Generate files of the mean face, the mean face plus first principal
component ... and so on up to the mean face plus all principal components. (Note the function SaveEigenface
normalises the eigenface before writing an image file so the mean face that you generate will have more contrast
than the input file MeanImage.jpg)
## Task 4.6
Complete the function PrincipalComponents which performs pca on a data set. To do this you will need to
use the Kohonen Lowe method. Your function should return the orthonormal basis as a list of eigenfaces
equivalent to the one returned by ReadEigenfaceBasis(). Check out the function numpy.linalg.eig before you
try to compute the eigenvectors yourself!
To check out your Principal components you should use the face images a.pgm to f.pgm. You can create an
image data set using the IDAPICourseworkLibrary, for example: imageData = array(ReadImages()). You can
find the mean image from your solution to 4.1.
## Results
Create a results file as before with the following items.
1. A title giving your group members
2. The mean vector Hepatitis C data set
3. The covariance matrix of the Hepatitis C data set
4. The component magnitudes for image “c.pgm” in the principal component basis used in task 4.4.
