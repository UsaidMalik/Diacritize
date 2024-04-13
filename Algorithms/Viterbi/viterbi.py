import copy
import sys
import pprint

class ViterbiAlgorithm():
    def __init__(self, ngram) -> None:
        self.wordLikelihood = {}
        self.transitionMatrix = {"Begin_Sent": {}, "End_Sent": {}}
        self.ngram = ngram
        
    def _normalizeProbabilities(self, matrix):
        for aRow in matrix:
            count = sum(matrix[aRow].values())
            for elem in matrix[aRow]:
                matrix[aRow][elem] /= count
        return matrix


    def trainViterbi(self, filePath):
        # essentially training the viterbi algorithm with the training file

        with open(filePath, "r", encoding="utf8") as trainingTxt:
            prevDiacritic = "Begin_Sent" # this is the previous diacritic which is begin sent in this case
            # this could possibly be changed to begin word?
            # i dont think thats a good idea tho since word inflection depends on other words as well as letters
            row = trainingTxt.readline() # starting seed value

            while row:
                if row.strip() == "ENDAYAT": # if i have reached the end of the sentence i need to do the new stuff
                    # checking to see if the previous POS exists in the transition matrix
                    self.transitionMatrix.setdefault(prevDiacritic, {}).setdefault("End_Sent", 0)
                    # the transition matrix gets the prev diacritic and the transition state from that value
                    self.transitionMatrix[prevDiacritic]["End_Sent"] += 1
                    # the transitions are updated but this doesnt really help tbh
                    prevDiacritic = "Begin_Sent"
                    # back to begin sent
                    # the the prev diacritic is begin sent
                    row = trainingTxt.readline() # got my new row
                    continue

                letter, Diacritic = "", ""
                if row.strip() != "":
                    letter, Diacritic =  row.split("\t")
                    # making sure row isnt a new line again
                letter = letter.strip()
                # updating word likelihoood matrix
                Diacritic = Diacritic.strip()

                self.wordLikelihood.setdefault(Diacritic, {}).setdefault(letter, 0)
                self.wordLikelihood[Diacritic][letter] += 1
                # incrementing the letter
                self.transitionMatrix.setdefault(prevDiacritic, {}).setdefault(Diacritic, 0)
                self.transitionMatrix[prevDiacritic][Diacritic] += 1

                prevDiacritic = Diacritic
                row = trainingTxt.readline()
            
        self.wordLikeLihood = self._normalizeProbabilities(self.wordLikelihood) # normalzing everything
        self.transitionMatrix = self._normalizeProbabilities(self.transitionMatrix) # normalizing everything to probabilities

    def _createStateIndices(self, transitionMatrix):
        stateIndices = {}
        states = list(transitionMatrix.keys())

        for i, state in enumerate(states):
            stateIndices[i] = state
        return stateIndices
        # initializing state indices
    
    def _printViterbi(self, viterbiMatrix):
        for row in viterbiMatrix:
            print(row)

    def readTestFile(self, devPath):
        sentencesArray = []

        with open(devPath, "r", encoding="utf-8") as devFile:

            tokens = []
            token = 1
            while token:
                token = devFile.readline()
                if token.strip() == "":
                    sentencesArray.append(tokens)
                    tokens = []
                else:
                    tokens.append(token.strip())
        return sentencesArray


    def _find_best_path(self, backpointer, stateIndices):
        # Start with the state that has the highest probability at the end of the sequence
        bestItemEnd = backpointer[1][-1] # this is the end column bestItem
        bestPath = [stateIndices[bestItemEnd]]

        currRowPtr = backpointer[bestItemEnd]
        for t in range(len(backpointer[0]) - 2, 1, -1):
            bestPath.append(stateIndices[currRowPtr[t]])
            currRowPtr = backpointer[currRowPtr[t]]

        bestPath.reverse()
        return bestPath
    

    def runViterbiAlgorithm(self, testFilePath, resultsPath):
        sentencesArray = self.readTestFile(testFilePath)
        # gets me all the sentences in the test file
        print(sentencesArray)
        stateIndices = self._createStateIndices(self.transitionMatrix)
        # gets me all the indexes of the states
        states = list(self.transitionMatrix.keys())
        # gets me all the states
        writeFile = open(resultsPath, "w", encoding="utf-8")
        # the file to write the results to 
        unknownWordProb = 1/500000

        for sentence in sentencesArray: # going through a single sentence

            # initializing viterbi
            firstLetter = ''
            try:
                firstLetter = sentence[0]
            except IndexError:
                print("Exception occured indexs error in the sntence")
                print("Sentence was ", sentence)
            # the first letter in the sentence that will be used to see the transition from beginsent to that letter

            viterbiMatrix = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
            # creating the matrix with the states
            backpointer = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(states))]
            # the backpointer matrix to find the best path
            viterbiMatrix[0][0] = 1
            # the probability of starting will always be one, technically i could make this a big number and it wouldnt matter
            # i might try that


            # this block finds the probabilities of going from the beginning of a sentence to 
            # the next particular state and letter combo
            for index in list(stateIndices.keys()):
                state = stateIndices[index]
                # this is the state represented by that index

                # the probability of that state going from beginning to that letter 
                viterbiMatrix[index][1] = self.transitionMatrix["Begin_Sent"].get(state, 0) * self.wordLikeLihood.get(state, {}).get(firstLetter, unknownWordProb)
               
            # done init viterbi for the begin sentences

            # this block fills in the rest of the viterbi algorithm
            # this is where the n-gram can be done
            for j in range(1, len(sentence)):
                # going through the sentence aka the words
                # the letter in the currente viterbi algo
                letter = sentence[j]
                for i, state in enumerate(states):
                    # the rows
                    maxProb = 0
                    # finding value that maximizes the next state
                    # finding prev state that maximizes
                    for k, prevState in enumerate(states): # going through all previous states
                        transitionProb = self.transitionMatrix.get(prevState, {}).get(state, 0)
                        # i will find the prevN prob of a word showing up. 

                        # here is the previous probability implemented as an ngram model focusing on the previous states
                        prevProb = viterbiMatrix[k][j]
                        currProb = prevProb * transitionProb
                        
                        # here is the curr probability that may be added to viterbi depending on its value
                        if currProb > maxProb:
                            maxProb = currProb
                            backpointer[i][j + 1] = k
                            # adding in the max possible probability to the backpoitner
                    viterbiMatrix[i][j + 1] = maxProb * self.wordLikeLihood.get(state, {}).get(letter, unknownWordProb)
                    # adding in the transition probabiltiy alongside the viterbi

            # capturing end state here for the backpointer matrix alongisde the n-grams
            for i, state in enumerate(states):
                maxProb = 0
                for k, prevState in enumerate(states):  # going through all previous states
                    transitionProb = self.transitionMatrix.get(prevState, {}).get(state, 0)
                    
                    lastIdx = len(sentence)

                    prevProb = viterbiMatrix[k][len(sentence)] 
                    currProb = prevProb * transitionProb
                    
                    # the previous N probabilties 
                    if currProb > maxProb:
                        maxProb = currProb
                        backpointer[i][lastIdx + 1] = k
                    viterbiMatrix[i][lastIdx + 1] = maxProb
            # done capturing end state
            self._printViterbi(viterbiMatrix)
            self._printViterbi(backpointer)
            bestPath = self._find_best_path(backpointer, stateIndices)
            for i in range(len(sentence)):
                writeFile.write(sentence[i] + "\t" + bestPath[i] + "\n")
            writeFile.write("\n")
            # writing to the results file by getting the best path 


        writeFile.close()