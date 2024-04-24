from pyarabic import araby
import sys
# documentation https://github.com/linuxscout/pyarabic/blob/master/doc/features.md

trainingFilePath = sys.argv[2]
originalDataPath = sys.argv[1]
trainingfile = open(trainingFilePath, "w", encoding='utf-8')
with open(originalDataPath, "r", encoding="utf-8") as Quran:
    for line in Quran:
        letters, marks = araby.separate(line)
        for idx, letter in enumerate(letters):
            trainingfile.write(f'{letter}\t{marks[idx]}\n')
        trainingfile.write("\n")