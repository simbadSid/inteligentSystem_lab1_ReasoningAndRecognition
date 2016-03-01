import os
import math



DEFAULT_COMMENT = "#"



def isEndOfFile(file):
    return (file.tell() == os.fstat(file.fileno()).st_size)

def nextMeaningLine(file, commentString=DEFAULT_COMMENT):
    while (not isEndOfFile(file)):
        res = file.readline().strip()
        if (res.startswith(commentString)):
            continue
        elif (res == "\n" or res == ""):
            continue
        else:
            return res
    raise Exception("No useful string found in the file " + file.name)

def vectorLength(vector):
    res = 0
    for i in xrange(len(vector)):
        res += math.pow(vector[i], 2)
    return math.sqrt(res)


def lowestValue(a, b):
    if (a < b):
        return a
    else:
        return b

def biggestValue(a, b):
    if (a > b):
        return a
    else:
        return b
