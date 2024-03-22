#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # break the phrase intro words
    words = x.split()
    # return the counter of each word
    return dict(Counter(words))

    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE

    def predict(x):
        score = dotProduct(featureExtractor(x), weights)
        if (score >= 0):
            return 1
        else:
            return -1 

    # looping through epochs to update the gradient descent & apply step size to define new w
    for a in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            if y*dotProduct(weights, phi) < 1:
                increment(weights, eta*y, phi)

        # print training and validaiton error
        
        trainError = evaluatePredictor(trainExamples, predict)
        validationError = evaluatePredictor(validationExamples, predict)
        print(f"Epoch {a}, Training error: {trainError}, Validation error: {validationError}")  

    # END_YOUR_CODE
    return weights



############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # note that there is intentionally flexibility in how you define phi.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        #gen feature vector with random numbers
        phi = {key: random.randint(1, 100) for key in weights}
        
        y=-1
        if dotProduct(phi, weights) >= 0:
            y = 1
        
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        #remove spaces/tabs
        x = x.replace(" ", "").replace("\t", "")
        features = {}
        #iterate on the string, extractong combinations
        for i in range(len(x) - n+1):
            combinations = x[i:i+n]
            #create combination or add 1
            if combinations in features:
                features[combinations]+=1
            else:
                features[combinations]=1
        return features
    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)

    # make first clusters in a randomized fashion 
    centroids = random.sample(examples, K) 
    assignments = [-1]*len(examples)
    # calculate before the squares
    exampleMagnitudes = [dotProduct(example, example) for example in examples]
    

    for epoch in range(maxEpochs):
        newAssignments = []
        loss = 0 
        
        # caculate the squared magnitudes for centroids
        centroidMagnitudes = [dotProduct(centroid, centroid) for centroid in centroids]


        # assign the points to centroid 
        for i, example in enumerate(examples):
            # calculate distance
            distances = [exampleMagnitudes[i] + centroidMagnitudes[k] - 2*dotProduct(example, centroids[k]) for k in range(K)]
            
            # Assign example to the nearest centroid
            assignment = distances.index(min(distances))
            newAssignments.append(assignment)
            loss+= min(distances)

        # check if need to stop 
        if newAssignments == assignments:
            break
        assignments = newAssignments


        # Recalculate centroids
        centroidSums = [{} for _ in range(K)]
        centroidCounts = [0]*K 
        for i, example in enumerate(examples):
            increment(centroidSums[assignments[i]], 1, example) 
            centroidCounts[assignments[i]]+=1
        # averaging the features to calculate it
        for k in range(K):
            for feature in centroidSums[k]:
                centroidSums[k][feature] /= centroidCounts[k] 
            centroids[k] = centroidSums[k]

    return centroids, assignments, loss

    # END_YOUR_CODE
