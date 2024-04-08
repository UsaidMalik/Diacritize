
def dataSplitter(dataPath, percentTrain, percentDev, percentTest):
    # takes data in the form of the raw ayat numbers 
    # so no side by side thing 
    with open(dataPath, "r", encoding="utf-8"):
        pass
    # will work on later

def cleanTestFile(testFilePath):
    writeFile = open("cleanedTest.txt", "w", encoding="utf-8")
    with open(testFilePath, "r", encoding="utf-8") as TxtFile:
        for line in TxtFile:
            if line == "\n":
                writeFile.write("\n")
            else:
                writeFile.write(line.rstrip().split("\t")[0] + "\n")


cleanTestFile("../Data/Parsed Data/test-test-set.txt")
