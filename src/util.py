import os




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
    raise Exception("No usefull string found in the file " + file.name)