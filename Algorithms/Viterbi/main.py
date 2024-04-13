from viterbi import ViterbiAlgorithm

viterbi = ViterbiAlgorithm(2)

viterbi.trainViterbi("../../Data/Parsed Data/training.txt")
viterbi.runViterbiAlgorithm("../../Data/Parsed Data/single-line.txt", "../../Data/Parsed Data/diacritized.txt")
