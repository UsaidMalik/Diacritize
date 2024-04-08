from viterbi import ViterbiAlgorithm

viterbi = ViterbiAlgorithm(2)

viterbi.trainViterbi("../../Data/Parsed Data/test-training-set.txt")
viterbi.runViterbiAlgorithm("../../Data/Parsed Data/cleanedTest.txt", "../../Data/Parsed Data/diacritized.txt")
