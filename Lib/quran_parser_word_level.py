from pyarabic import araby
import sys
# documentation https://github.com/linuxscout/pyarabic/blob/master/doc/features.md

trainingFilePath = sys.argv[2]
originalDataPath = sys.argv[1]
trainingfile = open(trainingFilePath, "w", encoding='utf-8')
with open(originalDataPath, "r", encoding="utf-8") as Quran:
    for line in Quran:
        words = line.split(" ")
        for word in words:
            letters, marks = araby.separate(word.strip())
            # trimming letters
            cleanedMarks = []
            trainingfile.write(letters) # write in the new letter
            trainingfile.write('\t')

            # this here is to clean up the marks 
            for i, mark in enumerate(marks):
                if mark ==  "ـ":
                    continue
                if mark == "ـ":
                    mark = ""
                cleanedMarks.append(mark)
                
            for cleanMark in reversed(cleanedMarks):
                trainingfile.write(" " + cleanMark)
            trainingfile.write("\n")

        trainingfile.write("ENDAYAT\n") # finishsed a line
