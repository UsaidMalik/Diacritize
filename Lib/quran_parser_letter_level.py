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
            cleanedLetters = []
            cleanedMarks = []
            for i, (letter, mark) in enumerate(zip(letters, marks)):
                if letter == "ـ" and mark ==  "ـ":
                    continue
                if mark == "ـ":
                    mark = ""
                cleanedLetters.append(letter)
                cleanedMarks.append(mark)

            idx = 0 
            while idx < len(cleanedLetters):
                letter = cleanedLetters[idx]
                mark = cleanedMarks[idx]
                if letter == "ّ":
                    cleanedMarks[idx - 1] = cleanedMarks[idx - 1] + " " + letter + " " + mark
                    cleanedMarks.pop(idx)
                    cleanedLetters.pop(idx)
                elif letter == "ٰ":
                    cleanedMarks[idx - 1] = cleanedMarks[idx - 1] + " " + letter 
                    cleanedMarks.pop(idx)
                    cleanedLetters.pop(idx)
                idx += 1
            
            for letter, mark in zip(cleanedLetters, cleanedMarks):
                 trainingfile.write(letter) # write in the new letter
                 trainingfile.write('\t')
                 trainingfile.write(mark)
                 trainingfile.write("\n")

            trainingfile.write("\n")
        trainingfile.write("ENDAYAT\n") # finishsed a line
