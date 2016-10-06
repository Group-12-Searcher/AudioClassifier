from PIL import Image, ImageTk

def convertGray(filename):
    im = Image.open(filename)
    im = im.convert('L')
    nameWithoutExt = filename.split(".")[0]
    parts = nameWithoutExt.split("\\")
    nameWithoutFilePath = parts[len(parts) - 1]
    im.save("sift\\" + nameWithoutFilePath + ".pgm")

def convertQueryToGray(filename):
    im = Image.open(filename)
    im = im.convert('L')
    nameWithoutExt = filename.split(".")[0]
    parts = nameWithoutExt.split("\\")
    nameWithoutFilePath = parts[len(parts) - 1]
    im.save("sift/temp/query.pgm")
