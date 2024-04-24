from unigramviterbi import UnigramViterbiAlgorithm
from viterbi import ViterbiAlgorithm

viterbiNGram = ViterbiAlgorithm(1)

viterbiNGram.trainViterbi("../../Data/Parsed Data/single-word-training.txt")

viterbiNGram.runViterbiAlgorithm("../../Data/Parsed Data/ngramtest.txt", "../../Data/Parsed Data/diacritizedNgram.txt")
