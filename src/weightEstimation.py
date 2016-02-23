from ProblemInstance import ProblemInstance
import matplotlib.pyplot as plt
import math



def improveModelParameter(modelParameter, trainingRate, pbInstance):
    resModelParameter = [0.0 for i in xrange(len(modelParameter))]
    alphaOverM = trainingRate / pbInstance.getNbrSample()
    for d in xrange(len(modelParameter)):
        if (d == 0):
            coeff = [1.0 for i in xrange(pbInstance.getNbrSample())]
        else:
            coeff = pbInstance.getFeatureAtDim(d-1)
        sum = pbInstance.computeWeightedLossSum(modelParameter, coeff)
        resModelParameter[d] = modelParameter[d] - alphaOverM*sum

    return resModelParameter




if __name__ == "__main__":
    pbInstance      = ProblemInstance()
    pbInstance.parseProblemInstance(inputFileName = "resource/input/parameter.txt")
    M2              = 2 * pbInstance.getNbrSample()
    axis            = [i for i in xrange(pbInstance.nbrIteraton)]

    plt.figure()

    for rate in pbInstance.getTrainingRate():
        modelParameter  = [0.0 for i in xrange(1+pbInstance.getFeatureDimension())]
        deltaLoss       = [0.0 for i in xrange(pbInstance.nbrIteraton)]
        l0              = pbInstance.computeSquareLossSum(modelParameter) / M2
        minDelta        = -1
        minW            = None
        isStrictDecrease= True
        for i in xrange(pbInstance.nbrIteraton):
            modelParameter  = improveModelParameter(modelParameter, rate, pbInstance)
            l1              = pbInstance.computeSquareLossSum(modelParameter) / M2
            deltaLoss[i]    = l1 - l0
            l0              = l1
            if (deltaLoss[i] > 0):
                isStrictDecrease = False
            if ((i == 0) or (deltaLoss[i] < minDelta)):
                minDelta    = deltaLoss[i]
                minW        = modelParameter
        plt.plot(axis, deltaLoss, label='Delta Loss (rate = ' + str(rate) + ')')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        print "alpha                = " + str(rate)
        print "Optimal W            = " + str(minW)
        print "Smallest deltaLoss   = " + str(deltaLoss[len(deltaLoss)-1])
        print "Is monotone decrease = " + str(isStrictDecrease)
        print "------------------"


