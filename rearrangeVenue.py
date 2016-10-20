fileIn = open("vine-venue-validation.txt", "rb")
audioVenue = {}

for line in fileIn:
    lineSplit = line.split("\t")
    audioVenue[lineSplit[0]] = lineSplit[1]

fileIn.close()
fileOut = open("vine-venue-validation-orderByName.txt", "wb")

for key in sorted(audioVenue.iterkeys()):
    line = key + "\t" + audioVenue[key]
    fileOut.write(line)

fileOut.close()
