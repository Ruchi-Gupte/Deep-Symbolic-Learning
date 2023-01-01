# Deep-Symbolic-Learning
This project explores an intuitive unsupervised learning algorithm to accurately detect communities in a given image dataset using symbolic Identification (ID) assignment. 

Deep Symbolic Learning seeks to incorporate the intuitive nature of SIFT with the efficiency of modern unsupervised learning which could help us get a process which
focuses on both results and execution for pattern recognition and feature detection. This is done using Co-Occurrence analytics, an intuitive algorithm which helps detect unknown patterns in unlabelled datasets. It does this by finding the possible co-occurrences of two given entities and finding the frequency of such entities with respect to other features. 


## Stage 1: Low-level Feature Extraction
An unlabelled image dataset is obtained and put into the system for low-level feature extraction. Low-level feature extraction deals with obtaining relevant information from the images, which contain even the slightest amount of detail pertaining to the given dataset.

![alt text](https://github.com/Ruchi-Gupte/Deep-Symbolic-Learning/blob/master/utils/Stage1.png)


## Stage 2: Feature Scoring

After low-level feature extraction, the feature scorer accepts the various low-level features and performs loose clustering on them to form clusters of similar features. For this, clustering techniques like K-Means, Mean-shift or Spectral Clustering can be used as they all club various features based on their location to the nearest cluster centroid. After clustering, each cluster is given its own unique identification (ID) and each feature under that cluster is labelled as the unique ID which symbolised that cluster, hence "symbolic ID"

![alt text](https://github.com/Ruchi-Gupte/Deep-Symbolic-Learning/blob/master/utils/Stage2.png)

 ### Hardcoded Directional Matrix
To determine the co-occurrence between two points or features in an image, the spatial location of each point is necessary. It is important to extract and determine if a point lies to the north, south, east or west to the other and by what degree. To do this, calculating the spatial location of each feature in every image is not only tedious but also unnecessary.

## Stage 3: Patch wise co-occurrence
Each symbolic ID matrix, representing an image, is superimposed on the hard-coded directional matrix and the symbolic ID pairs along with their directional values are
appended onto their respective tables directional tables. After processing every symbolic ID matrix, each table is further compressed and refined by aggregating the respective directional component of each unique pair in the matrix. 

![alt text](https://github.com/Ruchi-Gupte/Deep-Symbolic-Learning/blob/master/utils/Stage3.png)

## Stage 4 Community Detection
For the process of Community Detection, the algorithm starts with the maximum number of features interconnected together. A placeholder value is kept indicating the consistency of this community as a whole. Then the soft maximal clique is found by dropping one feature at a time in an attempt to increase the consistency.

## Stage 5: Community Modelling Stage
At this stage, the given features detected in the community detection stage are analysed and complied to detect the effectiveness and accuracy of the given system. The communities are structured and stored and the final list of features and communities are presented as an output to the system.


# Ending Remarks
The model is tested for its scalability and accuracy with conventional learning algorithms such as to test whether it can adequately detect useful features from a seemingly random and unlabelled dataset.
