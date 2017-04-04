#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for i in theData:
        prior[i[root]] += 1/float(len(theData))
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    for row in theData:
        cPT[row[varC], row[varP]] += 1
    for i in cPT.transpose():
        total = sum(i)
        if total != 0:
            i /= total
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    for i in theData:
        jPT[i[varRow]][i[varCol]] += 1/float(len(theData))    
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    for i in aJPT.transpose():
        total = sum(i)
        if total != 0:
            i /= total
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    for i in xrange(len(naiveBayes[0])):
        rootPdf[i] = naiveBayes[0][i]
        for x, y in enumerate(naiveBayes[1:]):
            a=theQuery[x]
            rootPdf[i] *= y[a, i]
    if(sum(rootPdf)!=0):
        rootPdf /= sum(rootPdf)
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    jP2=jP.transpose()
    for i in xrange(len(jP)):
        for j in xrange(len(jP[i])):
            if jP[i][j] != 0:
                mi += jP[i][j] * log2(jP[i][j]/(sum(jP[i])*sum(jP2[j])))
# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for i in xrange(len(MIMatrix)):
        for j in xrange(len(MIMatrix[i])):
            MIMatrix[i][j] = MutualInformation(JPT(theData, i, j, noStates))
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    for i in xrange(len(depMatrix)):
        for j in range(i+1, len(depMatrix[i])):
            depList.append((depMatrix[i][j], i, j))
    depList2 = sorted(depList, reverse=True)
# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
def attached(spanningTree,first,second,connection):
    for (dependancy,node1,node2) in spanningTree:
        if((node1==first and node2==second) or (node1==second and node2==first)):
            return True
        elif node1 == first and node2 not in connection:
            connection.append(node2)
            if attached(spanningTree, node2, second, connection):
                return True
        elif node2 == first and node1 not in connection:
            connection.append(node1)
            if attached(spanningTree, node1, second, connection):
                return True
    return False
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    for (dependency,node1,node2) in depList:
        if not attached(spanningTree,node1,node2,[]):
            spanningTree.append((dependency,node1,node2))
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
    for row in theData:
        cPT[row[child], row[parent1], row[parent2]] += 1
    for p1 in xrange(noStates[parent1]):
        for p2 in xrange(noStates[parent2]):
            total=0
            for cpt in xrange(len(cPT)):
                total += cPT[cpt][p1][p2]
            for cpt in xrange(len(cPT)):
                if total != 0:
                    cPT[cpt][p1][p2] = cPT[cpt][p1][p2] / total   
# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList

def BayesianNetwork(theData, noStates):
    arcList = [[0], [1], [2,0], [3,4], [4,1], [5,4], [6, 1], [7, 0, 1], [8, 7]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 0, 1, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]  
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for arc in arcList:
        if len(arc)==1:
            mdlSize +=  noStates[arc[0]]-1
        elif len(arc)==2:
            mdlSize +=  (noStates[arc[0]]-1)*noStates[arc[1]]
        elif len(arc)==3:
            mdlSize +=  (noStates[arc[0]]-1)*noStates[arc[1]]*noStates[arc[2]]
    mdlSize *= log2(noDataPoints) / 2
# Coursework 3 task 3 ends here 
    return mdlSize
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for i in xrange(len(dataPoint)):
        cpt = cptList[i]
        arc = arcList[i]
        if len(arc) == 1:
            if cpt[dataPoint[i]] != 0:
                jP *= cpt[dataPoint[i]]
        elif len(arc) == 2:
            if cpt[dataPoint[i]][dataPoint[arc[1]]] != 0:
                jP *= cpt[dataPoint[i]][dataPoint[arc[1]]]
        elif len(arc) == 3:
            if cpt[dataPoint[i]][dataPoint[arc[1]]][dataPoint[arc[2]]] != 0:
                jP *= cpt[dataPoint[i]][dataPoint[arc[1]]][dataPoint[arc[2]]]
# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for d in theData:
        mdlAccuracy += log2(JointProbability(d, arcList, cptList))
# Coursework 3 task 5 ends here 
    return mdlAccuracy

# Coursework 3 task 6 begins here
def BestScore(theData, arcList, cptList, noDataPoints, noStates):
    minimum = 100000.0
    x=-1
    y=-1
    for arcs in arcList:
        cpts=cptList[:]
        for arc in arcs[1:]:
            arcs.remove(arc)
            cpts.pop(arcs[0])
            if len(arcs) == 1:
                cpt = Prior(theData, arcs[0], noStates)
            elif len(arcs) == 2:
                cpt = CPT(theData, arcs[0], arcs[1], noStates)
            cpts.insert(arcs[0],cpt)
            mdlsize = MDLSize(arcList, cpts, noDataPoints, noStates)
            mdlaccuracy = MDLAccuracy(theData, arcList, cpts)
            mdlscore=mdlsize-mdlaccuracy
            if mdlscore < minimum:
                minimum = mdlscore
                y=arcs[0]
                x=arc
            arcs.append(arc)
            print y,x
    return minimum
# Coursework 3 task 6 ends here 
#
# End of coursework 3
#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
'''noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("IDAPIResults01.txt","Coursework One Results by Pavlos Siaperas")
AppendString("IDAPIResults01.txt","") #blank line
AppendString("IDAPIResults01.txt","The prior probability distribution of node 0 in the data set")
prior = Prior(theData, 0, noStates)
AppendList("IDAPIResults01.txt", prior)
AppendString("IDAPIResults01.txt","The conditional probability matrix P(2|0) calculated from the data.")
cpt = CPT(theData, 2, 0, noStates)
AppendArray("IDAPIResults01.txt", cpt)
AppendString("IDAPIResults01.txt","The joint probability matrix P(2&0) calculated from the data.")
jpt = JPT(theData, 2, 0, noStates)
AppendArray("IDAPIResults01.txt", jpt)
AppendString("IDAPIResults01.txt","The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0).")
jpt2cpt = JPT2CPT(jpt)
AppendArray("IDAPIResults01.txt", jpt2cpt)

cpt1 = CPT(theData, 1, 0, noStates)
cpt2 = CPT(theData, 2, 0, noStates)
cpt3 = CPT(theData, 3, 0, noStates)
cpt4 = CPT(theData, 4, 0, noStates)
cpt5 = CPT(theData, 5, 0, noStates)

AppendString("IDAPIResults01.txt","The results of queries [4,0,0,0,5] and [6, 5, 2, 5, 5] on the naive network")
query1 = Query([4,0,0,0,5], [prior, cpt1, cpt2, cpt3, cpt4, cpt5])
query2 = Query([6,5,2,5,5], [prior, cpt1, cpt2, cpt3, cpt4, cpt5])
AppendList("IDAPIResults01.txt", query1)
AppendList("IDAPIResults01.txt", query2)'''

#
# main program part for Coursework 2
#
'''noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults02.txt","Coursework Two Results by Pavlos Siaperas")
AppendString("IDAPIResults02.txt","")
AppendString("IDAPIResults02.txt","The dependency matrix for the HepatitisC data set")
depmatrix =DependencyMatrix(theData, noVariables, noStates)
AppendArray("IDAPIResults02.txt", depmatrix)
AppendString("IDAPIResults02.txt","The dependency list for the HepatitisC data set.")
deplist = DependencyList(depmatrix)
AppendArray("IDAPIResults02.txt", deplist)
AppendString("IDAPIResults02.txt","The spanning tree found for the HepatitisC data set")
tree=SpanningTreeAlgorithm(deplist,noVariables)
AppendArray("IDAPIResults02.txt", tree)'''
#
# main program part for Coursework 3
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults03.txt","Coursework Three Results by Pavlos Siaperas")
arcList, cptList = BayesianNetwork(theData, noStates)
AppendString("IDAPIResults03.txt","The MDLSize of the your network for Hepatitis C data set")
mdlsize = MDLSize(arcList, cptList, noDataPoints, noStates)
AppendString("IDAPIResults03.txt",mdlsize)
AppendString("IDAPIResults03.txt","The MDLAccuracy of the your network for Hepatitis C data set")
mdlaccuracy = MDLAccuracy(theData, arcList, cptList)
AppendString("IDAPIResults03.txt",mdlaccuracy)
AppendString("IDAPIResults03.txt","The MDLScore of the your network for Hepatitis C data set")
mdlscore=mdlsize-mdlaccuracy
AppendString("IDAPIResults03.txt",mdlscore)
AppendString("IDAPIResults03.txt","The score of your best network with one arc removed")
bestscore=BestScore(theData, arcList, cptList, noDataPoints, noStates)
AppendString("IDAPIResults03.txt",bestscore)
