# K-NN-Implementation

# Abstract:
 The nearest-neighbor algorithm is simple. It is intuitive, accurate and applicable to various problems. The simple form is l-NN algorithm assigns an unknown input sample to the category of its nearest neighbor from a stored labeled reference set. The k-NN algorithm looks at k samples in the reference set that are closest to the unknown sample and carries out a vote to make a decision. Given a dataset of Ecoli, Yeast, Glass we predict the k-NN for sample test examples produced from K-fold cross validation and find the accuracy of each fold.

# Introduction:
 The k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. We use K-NN to classify the test samples. We start k with 1 and then to 2,3,5,10,12. The distance metric to calculate the distance between the training set and test set is Euclidean and the Kernel functions RBF and Polynomial. We use k-fold cross validation to split the sets to training and test and eventually calculate the accuracy of fold for a given K( k of KNN ) 

The Euclidean distance measure uses the formula = sqrroot (sum (trainset- testset ) ^2)
Kernel Polynomial = K(X, Y) = (gamma <X, Y> + coef0)^degree
Kernel RBF=K(x, y) = exp(-gamma ||x-y||^2)
The above formulas are used for the distant metric and the nearest k votes is chosen to classify the test sample
KNN Instance based classifier:
           KNN is a 
                         1) Instance based learning
                         2) Lazy learning algorithm
                         3) Commonly used distance metric for continuous variables is Euclidean distance
	                       4) Supervised leaning algorithm
The training examples are vectors in a multidimensional feature space, each with a class label. The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples.
We use the kernel functions Radial basis function and Polynomial kernel
RBF kernel: The function computes the radial basis function (RBF) kernel between two vectors. where x and y are the input vectors. If  the kernel is known as the Gaussian kernel of variance .
In the program is it 
 k_exp =2 - 2 * np.exp( k*np.linalg.norm(np.subtract(xtrain,xtest))/np.square(xtest.std()))

# Polynomial function:
 The function computes the degree-d polynomial kernel between two vectors. The polynomial kernel represents the similarity between two vectors. Conceptually, the polynomial kernels considers not only the similarity between vectors under the same dimension, but also across dimensions. When used in machine learning algorithms, this allows to account for feature interactionIn the program,
pow(gamma*np.dot(xtrain.T,xtrain) + delta,degree) + pow(gamma*np.dot(xtest.T,xtest) + delta,degree) - 2 * pow(gamma*np.dot(xtrain.T,xtest) + delta,degree)

Distance measure used is Euclidean distance (distance formula given above)

# Datasets:
 The datasets used are Ecoli, Glass and yeast
 
 # Ecoli:
    The objective of this problem is to predict the localization site of proteins by employing some measures about the cell (cytoplasm, inner membrane, perisplasm, outer membrane, outer membrane lipoprotein, inner membrane lipoprotein inner membrane, cleavable signal sequence). To assess the data to classification process, the first attribute of the original data set (the sequence name) has been removed in this version
    
    Attribute description:
      1. Sequence Name: Accession number for the SWISS-PROT database
      2. mcg: McGeoch's method for signal sequence recognition.
      3. gvh: von Heijne's method for signal sequence recognition.
      4. lip: von Heijne's Signal Peptidase II consensus sequence score.
      5. chg: Presence of charge on N-terminus of predicted lipoproteins.
      6. aac: score of discriminant analysis of the amino acid content 
      7. alm1: score of the ALOM membrane spanning region prediction program.
      8. alm2: score of ALOM program after excluding putative cleavable signal
	   
    Class Distribution:
      cp  (cytoplasm)                                    
      im  (inner membrane without signal sequence)                      
      pp  (perisplasm)                                   
      imU (inner membrane, uncleavable signal sequence)  
      om  (outer membrane)                                
      omL (outer membrane lipoprotein)                     
      imL (inner membrane lipoprotein)                     
      imS (inner membrane, cleavable signal sequence)  

# Glass:
 This data set contains the description of 214 fragments of glass originally collected for a study in the context of criminal investigation. Each fragment has a measured reflectivity index and chemical composition (weight percent of Na, Mg, Al, Si, K, Ca, Ba and Fe)

    Class Distribution: (out of 214 total instances)
      163 Window glass (building windows and vehicle windows), 87 float processed ,70 building windows,17 vehicle wind, 76 non-float process, 76 building windows, 0 vehicle wind, 51 Non-window glass, 13 container,  9 tablewa  ,29 headlamps

# Yeast:
      This database contains information about a set of Yeast cells. The task is to determine the localization site of each cell among 10 possible alternatives. 

    Attributes description: 
     1. Mcg: McGeoch's method for signal sequence recognition. 
     2. Gvh: von Heijne's method for signal sequence recognition. 
     3. Alm: Score of the ALOM membrane spanning region prediction program. 
     4. Mit: Score of discriminant analysis of the amino acid content 
     5. Erl: Presence of "HDEL" substring (thought to act as a signal for retention in the endoplasmic reticulum lumen). 
