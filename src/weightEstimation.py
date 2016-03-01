from ProblemInstance import ProblemInstance
import matplotlib.pyplot as plt
import math
from cupshelpers.ppds import normalize






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

def processOutput(rateList, minParameter, minLoss, minDeltaLoss, axis, loss, parameter0, parameter1, isStrictDecrease=True):
    # print the results
    print "\n"
    print "Used rates           = " + str(rateList)
    print "Optimal W            = " + "w0 = " + str(minParameter[0]) + "\t w1 = " + str(minParameter[1])
    print "Smallest loss        = " + str(minLoss)
    print "Smallest delta loss  = " + str(minDeltaLoss)
    print "Is monotone decrease = " + str(isStrictDecrease)
    print "------------------"

    # Print the loss curve
    plt.plot(axis, loss,    label='Loss (with rate list = ' + str(rateList) + ')')
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Print the parameter curve
    plt.plot(axis, parameter0, label='w0 (with rate list = ' + str(rateList) + ')')
    plt.plot(axis, parameter1, label='w1 (with rate list = ' + str(rateList) + ')')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pbInstance      = ProblemInstance()
    pbInstance.parseProblemInstance(normalize=False, inputFileName = "../resource/input/parameter.txt")
    M2              = 2 * pbInstance.getNbrSample()
    axis            = [i for i in xrange(pbInstance.nbrIteraton)]

    plt.figure()

    for rate in pbInstance.getTrainingRate():
        alpha           = rate
        alphaList       = []
        modelParameter  = [0.0 for i in xrange(1+pbInstance.getFeatureDimension())]
        loss            = [0.0 for i in xrange(pbInstance.nbrIteraton)]
        w0              = [0.0 for i in xrange(pbInstance.nbrIteraton)]
        w1              = [0.0 for i in xrange(pbInstance.nbrIteraton)]
        minLoss         = -1
        minDeltaLoss    = -1
        minW            = None
        isStrictDecrease= True
        alphaList.append(alpha)
        for i in xrange(pbInstance.nbrIteraton):
            modelParameter  = improveModelParameter(modelParameter, alpha, pbInstance)
            loss[i]         = pbInstance.computeSquareLossSum(modelParameter) / M2
            w0[i]           = modelParameter[0]
            w1[i]           = modelParameter[1]
            if ((i > 0) and (loss[i] > loss[i-1])):
                isStrictDecrease = False
                alpha /= 10
                i -= 1
                alphaList.append(alpha)
            else:
                minLoss     = loss[i]
                minW        = modelParameter
                minDeltaLoss=(loss[i-1] - loss[i])

        processOutput(alphaList, minW, minLoss, minDeltaLoss, axis, loss, w0, w1, isStrictDecrease)

