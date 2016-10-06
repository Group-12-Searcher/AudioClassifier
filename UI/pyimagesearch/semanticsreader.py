import csv

class SemanticsReader:
        def __init__(self):
                self

        def read(self, file):
                reader = csv.reader(open(file), delimiter=" ")
                data = []
                for line in reader: #the text file should only have 1 line
                        for i in range(len(line)-1):
                                data.append(float(line[i][1:]))
                return data


