from IndexedQueue import IndexableQueue
from typing import List, Dict, TypeVar

TransitionMatrix = TypeVar('TransitionMatrix', bound=Dict[str, Dict[str, int]])
WordLikelihood = TypeVar('WordLikelihood', bound=Dict[str, int])

class ViterbiAlgorithm():
    def __init__(self, ngram: int) -> None:
        self.ngram : int = ngram
        self.BEGIN_MACRO : str = "BEGIN_SENT"
        self.END_MACRO : str = "ENDAYAT"
        self.wordLikelihoods : List[WordLikelihood] = [{} for _ in range(self.ngram)]# this is the n gram model
        """
        Type of Above is : 

        WordLikelihoods : [
        {"fa": "1000", "ba":100 ...}, {"fa ba": 500, "alif ba": 50...}
        ]
        # These are the letter or word likelihoods dependant and are seperated by spaces as keys
        # there are n versions of these for however many n grams are present
        # it goes P(1 and ... and N) / P(1 and ... and N - 1) to get the probability of an N gram model
        for the word likelihoods 
        """
        self.transitionMatrixs : List[TransitionMatrix] = [{self.BEGIN_MACRO*i : {}, self.END_MACRO*i: {}} for i in range(self.ngram)] 
        """
        Type of Above is : 

        TransitionMatrixList : [
        {"BEGIN_SENT": {"Fatha": 50, "Dammah": 60, "Kasra": 70}...}, {"BEGIN_SENT Fatha": { Dammah Kasra" : "Fatha Dammah, Dammah Kasra..."}, ...}
        ]
        # that is how the N gram model works, the Transition matrix at index n - 1 is the nth  n-gram. 
        #it contains transitions of an  to n mapping
        the matrix as n transition matrices of type str : str
        """
        
    def _normalizeProbabilities(self, matrix):
        for aRow in matrix:
            count = sum(matrix[aRow].values())
            for elem in matrix[aRow]:
                matrix[aRow][elem] /= count
        return matrix


    def _updateMatrix(self, letterQueue, diacriticQueue, prevDiacriticQueue, i):

        # these three are keys used to queue into the matricies
        letterKeyArr : List[str]= []
        diacriticKeyArr : List[str] = []
        prevDiacriticKeyArr : List[str] = []

        for nletter, ndiacritic, nPrevDiacritic in zip(letterQueue, diacriticQueue, prevDiacriticQueue):
            letterKeyArr.append(nletter)
            diacriticKeyArr.append(ndiacritic)
            prevDiacriticKeyArr.append(nPrevDiacritic)
        
        letterKey = " ".join(letterKeyArr)
        diacriticKey = " ".join(diacriticKeyArr)
        prevdiacriticKey = " ".join(prevDiacriticKeyArr)

        self.wordLikelihoods[i].setdefault(diacriticKey, {}).setdefault(letterKey, 0)
        self.wordLikelihoods[i][diacriticKey][letterKey] += 1
        # incrementing the gram if found
        self.transitionMatrixs[i].setdefault(prevdiacriticKey, {}).setdefault(diacriticKey, 0)
        self.transitionMatrixs[i][prevdiacriticKey][diacriticKey] += 1

    def trainViterbi(self, filePath: str) -> None:
        # essentially training the viterbi algorithm with the training file
        # this trains viterbi aka fills in the word likelihood and transition matrices with their right values

        with open(filePath, "r", encoding="utf8") as trainingTxt:
            # this could possibly be changed to begin word?
            # i dont think thats a good idea tho since word inflection depends on other words as well as letters
            row : str = trainingTxt.readline() # starting seed value from the file
            
            letterNGram : IndexableQueue = IndexableQueue(maxsize=self.ngram) # these are all the n gram queues that are being used for the states of the letters acquired
            diacriticNGram : IndexableQueue= IndexableQueue(maxsize=self.ngram) # the states acquired so far
            prevDiacriticNGram : IndexableQueue = IndexableQueue(maxsize=self.ngram) # the previous states used for the transition matrix

            prevDiacriticNGram.put(self.BEGIN_MACRO) # pushing begin sent as the prevDiacritic to be put in

            # the way this will work is ill init this to be of size n_gram
            # this is basically just a queue that removes when too large
            while row:

                # if i've reached the end then its time ot dequeu eveything
                if row.strip() == self.END_MACRO: # if i have reached the end of the sentence i need to do the new stuff
                    # if i dont fill up before getting to the end its fine
                    # checking to see if the previous POS exists in the transition matrix
                    # finally since ive reached the end then that means that the queue is full and i need to add in all the items that were missed in teh queue
                    # [a, b, c] full queue and so far everything has been placed in except for the n - 1 grams and going forward in the queue

                    letterNGram.put("END_LETTER") # dummy letter
                    diacriticNGram.put(self.END_MACRO) # since i basically am at END
                    # dont know why but n^2 seems neccesary for getting all the last n grams out of the queue
                    for _ in range(self.ngram): # looping through the n gram
                        # first iter ill grab 0 - n -1
                        # then from 0 - n - 3
                        for i in range(len(letterNGram)): # ill go from the top 5 in the beginning
                            # only looping ngram - 1 times since all have basically been captured except the deeper n - 1 i can do the same thing but
                            # just pop from the queue this time to get those items
                            # so i need to keep popping off and then getting it into the matrices before adding the final transitions
                            # so ill have the proper queue here no. bcz it doesnt capture everyhting 

                            # need to do something like this.
                            nGramLetter = letterNGram[0:i + 1]
                            nGramDiacritic = diacriticNGram[0:i + 1]
                            nGramPrevDiacritic = prevDiacriticNGram[0:i + 1]
                            # since i have my letters and queus now i can add in teh keys 
                            self._updateMatrix(nGramLetter, nGramDiacritic, nGramPrevDiacritic, i)

                        letterNGram.pop() # popping the last one since it was already captured before entering here
                        diacriticNGram.pop()#
                        prevDiacriticNGram.pop() # these should be empty now
                    # this seems like itll work
                    
                    prevDiacriticNGram.put(self.BEGIN_MACRO) # basically new sentence starting up now
                    # back to begin sent
                    # the the prev diacritic is begin sent
                    row = trainingTxt.readline() # got my new row
                    continue # starting loop from beginning

                letter, Diacritic = "SPACE", "SPACE" # diacritic is nothing
                if row.strip() != "":
                    letter, Diacritic =  row.split("\t")
                    # making sure row isnt a new line again

                letter = letter.strip()
                Diacritic = Diacritic.strip()
                
                # ill append it in if it hasnt reached the large size yet
                letterNGram.put(letter)
                diacriticNGram.put(Diacritic) # adding in the elemnts to the queue
                # now i need to add in the N gram versions of the stuff into my matrices

                # [c, d, e]
                # wait for queue to fill up. stay in place capturing the elements
                # queuee fill up isnt guaranteed better to just capture on start
                # then at the end pop out the queue and have everyone capture until elements not 
                # once queue full stay in place
                # if queue empty again then capture what you missed
                # [d, e, ..] # or just pop everything out so it goes down
                if letterNGram.full():
                    # if i never fill up its fine when i reach the end marker it will do the missed capturing
                    # the thing is full so i can just pull my grams by indexing like [:]
                    for i in range(self.ngram):
                        # this here just gets the stuff to get the keys and used to create and update the matrix
                        nGramLetter : List[str] = letterNGram[0:i + 1]
                        nGramDiacritic : List[str] = diacriticNGram[0: i + 1]
                        nGramPrevDiacritic : List[str] = prevDiacriticNGram[0: i + 1]

                        self._updateMatrix(nGramLetter, nGramDiacritic, nGramPrevDiacritic, i)

                prevDiacriticNGram.put(Diacritic)
                # updating the prev diacritic in the prev diacritic n gram
                row = trainingTxt.readline()
                # grabbing next line
    
    def _getTransitionProb(self, i, prevState : str, currState: str):
        # i is the index of which n gram im getting 
        # key is the previous n - 1 states
        # curr item is the next state
        # prevKey is the previous n - 2 states
        # proba is P(a, b, c) / P(a, b)
        # for the transition matrix i find the probability of going from 
        # the previous n - 1 states to the next state 
        # for the n gram version ill just do that but with the prev 
        # for the n gram version its wahts the probability of the previous n - 2 states to the next state
        # basically the new item and the old item division as the probability
        # key is the states key so a, b, c
        # i is the n gram number so the nth gram or n 
        currKeyString = currState
        prevKeyString = prevState

        currStateArr = currState.split(" ")
        prevStateArr = prevState.split(" ")

        prevNMinusOneStatesArr = prevState.split(" ")[1:] # previous n - 1 states 
        currNMinusOneStatesArr = currState.split(" ")[1:]
        prevNMinusOneStatesKey = " ".join(prevNMinusOneStatesArr)
        currNMinusOneStatesKey = " ".join(currNMinusOneStatesArr)

        num = self.transitionMatrixs[i].get(prevKeyString, {}).get(currKeyString, 0)
        j = 0
        while num == 0 and i - j > 0:
             # if its zero perform backoff on the keys
            currKeyString = " ".join(currStateArr[j:])
            prevKeyString = " ".join(prevStateArr[j:])
            prevNMinusOneStatesKey = " ".join(prevNMinusOneStatesArr[j:])
            currNMinusOneStatesKey =  " ".join(currNMinusOneStatesArr[j:])

            num = self.transitionMatrixs[i - j].get(prevKeyString, {}).get(currKeyString, 0) # this is backing off to the previous number
            j += 1
  
        if (i - j) == 0:
            count = sum(self.transitionMatrixs[0].get(prevKeyString, {"": 1}).values())

            return self.transitionMatrixs[0].get(prevKeyString, {}).get(prevKeyString, 0)/count
        # its impossible for the curr key to exist and not the prev n key so if that is the case there is a bug
        else: # unknown word probability of 1/10000 if prev doesnt exist neither does curr
            return self.transitionMatrixs[i - j].get(prevKeyString, {}).get(prevKeyString, 0)/(self.transitionMatrixs[i - j - 1].get(prevNMinusOneStatesKey, {}).get(currNMinusOneStatesKey, 10000) + 1)

        

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
    
    def _getWordLikelihoodProb(self, i, currNStates : str, letterNGram : List[str]):
        # proba is P(a, b, c) / P(a, b)
        # basically the new item and the old item division as the probability
        # key is the states key so a, b, c
        # i is the n gram number so the nth gram or n 
        # given as a list of the previous numbers
        # this is just the curr item divided by prev item
        currKeyArr = currNStates.split(" ") # take everything but last
        currKeyString = currNStates

        nMinusOneKeyArr = currKeyArr[1:]
        nMinusOneKeyString = " ".join(nMinusOneKeyArr)

        letterNGramKey = " ".join(letterNGram)
        letterNGramKeyMinusOne = " ".join(letterNGram[1:])

        num = self.wordLikelihoods[i].get(currNStates, {}).get(letterNGramKey, 0)
        j = 0 # making zero so i could do a trick
        while num == 0 and i - j > 0: # making sure im not at the zeroeth nunber (a unigram)
            # if its zero perform backoff on the keys
            currKeyString = " ".join(currKeyArr[j:])
            nMinusOneKeyString = " ".join(nMinusOneKeyArr[j:])
            letterNGramKeyMinusOne = " ".join(letterNGramKey[j:])
            letterNGramKey =  " ".join(letterNGramKey[j:])

            num = self.wordLikelihoods[i - j].get(currKeyString, {}).get(letterNGramKey, 0) # this is backing off to the previous number
            j += 1

        if (i - j) == 0:
            count = sum(self.wordLikelihoods[0].get(currKeyString, {"": 1}).values())
            return self.wordLikelihoods[0].get(currKeyString, {}).get(letterNGramKey, 1/10000)/count
        # its impossible for the curr key to exist and not the prev n key so if that is the case there is a bug
        else: # unknown word probability of 1/10000 if prev doesnt exist neither does curr
            return self.wordLikelihoods[i - j].get(currKeyString, {}).get(letterNGramKey, 1/1000)/(self.wordLikelihoods[i - j - 1].get(nMinusOneKeyString, {}).get(letterNGramKeyMinusOne, 1/10000) + 1)

    def runViterbiAlgorithm(self, testFilePath, resultsPath):
        sentencesArray = self.readTestFile(testFilePath)
        # gets me all the sentences in the test file
        stateIndices = self._createStateIndices(self.transitionMatrixs[-1])
        # i should just use this tbf since it doenst matter
        # these are just indices
        # gets me all the indexes of the states for a unigram since thats how it gets asssigned
        nGramStates = list(self.transitionMatrixs[-1].keys())
        print(self.transitionMatrixs[-1])
        # get the states for the last one
        # i should just use this tbf since it doenst matter

        # gets me all the indexes of the states for a unigram since thats how it gets asssigned
        # i can get the first tranisiotn matrices keys
        # and use them as states or i can go to the other ones
        # idk how to do this for an n gram model?

        # gets me all the states
        writeFile = open(resultsPath, "w", encoding="utf-8")
        # the file to write the results to 
        unknownWordProb = 1/5000

        for sentence in sentencesArray: # going through a single sentence


            # the first letter in the sentence that will be used to see the transition from beginsent to that letter

            viterbiMatrix = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(nGramStates))]
            # creating the matrix with the states
            backpointer = [[0 for _ in range(len(sentence) + 2)] for _ in range(len(nGramStates))]
            viterbiMatrix[0][0] = 1
            # the backpointer matrix to find the best path
            # the probability of starting will always be one, technically i could make this a big number and it wouldnt matter
            # i might try that
            for index in list(stateIndices.keys()):
                state = stateIndices[index]
                # this is the state represented by that index

                # the probability of that state going from beginning to that letter 
                viterbiMatrix[index][1] = self.transitionMatrix["Begin_Sent"].get(state, 0) * self.wordLikeLihood.get(state, {}).get(firstLetter, unknownWordProb)

            # this block finds the probabilities of going from the beginning of a sentence to 
            # the next particular state and letter combo

            # this block fills in the rest of the viterbi algorithm
            # this is where the n-gram can be done
            # ill loop through my n gram states and get the highest probability given my n - 1letters then just do backoff
            # going through each letter in the sentence
            for j in range(0, len(sentence)):
                # going through the sentence aka the words
                # the letter in the currente viterbi algo

                letterPreviousNgram : List[str] = sentence[0:j]
                n = j # this is the amount of the n gram so how much of the back i look at

                if j - self.ngram  >= 0: # more words than n grams so good to go 
                    letterPreviousNgram = sentence[j-self.ngram:j] # the previous letter was the first in the sentence for the 1 gram model
                    n = self.ngram - 1 # at my total n gram

                # i think i have to do the n combos for the states here instead so i can loop through them and find the 
                # one with the biggest probability given my letter or my n - 1 letters
                nGramStates = list(self.transitionMatrixs[n].keys())
                # each possible state the letter may be in 
                for i, state in enumerate(nGramStates):
                    # the rows
                    maxProb = 0
                    # finding value that maximizes the next state
                    # finding prev state that maximizes
                    for k, prevState in enumerate(nGramStates): # going through all previous states
                        # i could probably drop in an n gram loop here actually to get the diff transition probs

                        # this is n gram ish version that odesnt exactly work
                        #for prevNState in nGramStates:
                        #    transitionProb = self._getTransitionProb(n, prevNState, state)

                        # i will find the prevN prob of a word showing up. 
                            # im using the first transition amtrix here 
                            transitionProb = self._getTransitionProb(n, prevState, state)
                        # here is the previous probability implemented as an ngram model focusing on the previous states
                            prevProb = viterbiMatrix[k][j]
                            currProb = prevProb * transitionProb
                            
                            # here is the curr probability that may be added to viterbi depending on its value
                            if currProb > maxProb:
                                maxProb = currProb
                                backpointer[i][j + 1] = k
                            # adding in the max possible probability to the backpoitner
                    viterbiMatrix[i][j + 1] = maxProb * self._getWordLikelihoodProb(n, state, letterPreviousNgram)
                    # adding in the transition probabiltiy alongside the viterbi

            # capturing end state here for the backpointer matrix alongisde the n-grams
            # done capturing end state

            bestPath = self._find_best_path(backpointer, stateIndices)
            for i in range(len(sentence)):
                writeFile.write(sentence[i] + "\t" + bestPath[i] + "\n")
            writeFile.write("\n")
            # writing to the results file by getting the best path 


        writeFile.close()