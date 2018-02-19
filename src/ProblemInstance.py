from array import array
from util import *





class ProblemInstance:
    # -----------------------------
    # Attributes
    # -----------------------------
    """
    nbrIteration
    trainingRate            = []        # Lecture notation: alpha
    trainingSample_feature  = [[]]      # Lecture notation: X:
    trainingSample_result   = [[]]      # Lecture notation: Y
    """

    # -----------------------------
    # Builder
    # -----------------------------
    def parseProblemInstance(self, inputFileName, normalize=True):
        file            = open(inputFileName)

        self.nbrIteraton= int(nextMeaningLine(file))
        # Init training rate
        nbrTrainingRate = int(nextMeaningLine(file))
        self.trainingRate = [0.0 for i in xrange(nbrTrainingRate)]
        for i in xrange(nbrTrainingRate):
            self.trainingRate[i] = float(nextMeaningLine(file))
        # Init feature samples
        nbrSamples          = int(nextMeaningLine(file))
        featureDimension    = int(nextMeaningLine(file))
        self.trainingSample_result  = [0.0 for i in xrange(nbrSamples)]
        self.trainingSample_feature = [[0.0 for i in xrange(featureDimension)] for j in xrange(nbrSamples)]
        for sample in xrange(nbrSamples):
            self.trainingSample_result[sample] = float(nextMeaningLine(file))
            for feature in xrange(featureDimension):
                self.trainingSample_feature[sample][feature] = float(nextMeaningLine(file))

        if (normalize == True):
            self.normalize()
        file.close()

    # -----------------------------
    # Getter
    # -----------------------------
    def getTrainingRate(self):
        return self.trainingRate

    def getNbrSample(self):
        return len(self.trainingSample_result)

    def getFeatureDimension(self):
        return len(self.trainingSample_feature[0])

    # return the set of the dim th coefficients of all the training features 
    def getFeatureAtDim(self, dim):
        res = [0.0 for i in xrange(self.getNbrSample())]
        for i in xrange(self.getNbrSample()):
            res[i] = self.trainingSample_feature[i][dim]
        return res
    # -----------------------------
    # Local methods
    # -----------------------------
    # -----------------------------
    # Estimate the output corresponding to the input feature using the given model
    # Supposes that the model is linear
    # -----------------------------
    def predictResult(self, modelParameter, featureIndex):
        feature     = self.trainingSample_feature[featureIndex]
        dimension   = len(feature)
        if (len(modelParameter) != (dimension+1)):
            raise Exception("The feature and the parameter of the model have different dimensions")

        res = modelParameter[0]
        for i in xrange(dimension):
            res += modelParameter[i+1] * feature[i]
        return res

    # -----------------------------
    # Compute the squared sum of the difference between the known output and
    # the output computed using the given model
    # -----------------------------
    def computeSquareLossSum(self, modelParameter):
        sum         = 0
        nbrSamples  = self.getNbrSample()
        for m in xrange(nbrSamples):
            scalar  = self.predictResult(modelParameter, m)
            scalar  -= self.trainingSample_result[m]
            scalar  *= scalar
            sum     += scalar
        return sum

    # -----------------------------
    # Compute the weighted sum of the difference between the known output and
    # the output computed using the given model
    # -----------------------------
    def computeWeightedLossSum(self, modelParameter, ponderation=None):
        if ((ponderation != None) and (len(ponderation) != self.getNbrSample())):
            raise Exception("The ponderation array size differs from the number of samples")

        sum = 0
        for m in xrange(self.getNbrSample()):
            scalar  = self.predictResult(modelParameter, m)
            scalar  -= self.trainingSample_result[m]
            if (ponderation != None):
                scalar *= ponderation[m]
            sum     += scalar 
        return sum


    def normalize(self):
        max = -1
        min = -1
        for sample in xrange(self.getNbrSample()):
            X       = self.trainingSample_feature[sample]
            length  = vectorLength(X)
            m0      = lowestValue(length, self.trainingSample_result[sample])
            m1      = biggestValue(length, self.trainingSample_result[sample])
            if ((sample == 0) or (m0 < min)):
                min = m0
            if ((sample == 0) or (m1 > max)):
                max = m1
        if (min == max):
            raise Exception ("Unhandled special case: min = max")
        r = max - min
        for sample in xrange(self.getNbrSample()):
            X = self.trainingSample_feature[sample]
            for j in xrange(len(X)):
                X[j] = X[j] / r
            self.trainingSample_feature[sample] = X



    def printProblemInstance(self):
        print ("Problem instance:\n")
        print ("\t- Number of iterations: " + int(self.nbrIteraton))
        msg = ""
        for i in xrange(len(self.trainingRate)):        # Print the training rate list
            if (i > 0):
                msg += ", "
            msg += str(self.trainingRate[i]) 
        print ("\t- Training rate       : " + msg)
        print ("\t- Number of samples   : " + str(self.getNbrSample()))
        print ("\t- Feature dimension   : " + str(self.getFeatureDimension()))
        print ("\t- training samples    : ")
        for sample in xrange(self.getNbrSample()):          # Print the training samples
            print ("\t\t y[" + str(sample) + "]\t = " + str(self.trainingSample_result[sample]))
            print ("\t\t X[" + str(sample) + "]\t = " + str(self.trainingSample_feature[sample]))
            print ("\t\t ------------------------------")
