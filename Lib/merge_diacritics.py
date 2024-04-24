from pyarabic import araby
import sys
# documentation https://github.com/linuxscout/pyarabic/blob/master/doc/features.md

diacriticsFilePath = sys.argv[1]
mergedFilePath = sys.argv[2]
mergedFile = open(mergedFilePath, "w", encoding='utf-8')
with open(diacriticsFilePath, "r", encoding="utf-8") as WordsFile:
    for entry in WordsFile:
        if entry == "ENDAYAT" or entry.strip() == "":
                mergedFile.write("END")
                continue
        
        word, diacritics = entry.strip().split("\t")     
        diacriticsArr = diacritics.split(" ")

        if len(word) == len(diacriticsArr):
             diacriticsArr = reversed(diacriticsArr)
        else:
            while len(diacriticsArr) < len(word):
                diacriticsArr.append(" ") # dummy diacritic
            while len(diacriticsArr) > len(word):
                diacriticsArr.pop() # remove a diacritic
            
        diacritics = "".join(diacriticsArr)
        # trimming letters
        joinedMarks = araby.joint(word, diacritics)
        mergedFile.write(joinedMarks)
        mergedFile.write("\n")
    
