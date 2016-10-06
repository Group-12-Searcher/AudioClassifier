import subprocess
import os

def convertKey(pgmFile):
    # This file must be run at directory ImageSeach_demo and not its
    # subdirectories.
    
    os.chdir("sift") # go into subdirectory sift

    fileName = pgmFile.replace("sift\\", "") # remove sift\ from pgmFile
    inputfile = open(fileName, "r") # read pgmFile
    outputFileName = fileName.split(".")[0] + ".key"   
    outputfile = open(outputFileName, "w") # write to .key file

    # Run the command prompt here.
    #siftWin32 < (target PGM file) > (destination .key file)
    subprocess.call("./siftWin32", stdin=inputfile, stdout =outputfile)

    # Close both files which are opened for reading and writing
    inputfile.close()
    outputfile.close()

    # switch directory back to avoid screwing up other codes
    os.chdir("../")
