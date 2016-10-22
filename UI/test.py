# import the necessary packages
import sys
import numpy as np
import os
os.chdir("../")
sys.path.append(os.getcwd())
from classifier.predict import predictSVM
os.chdir("UI")
import cv2
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk, ImageDraw, ImageFont


class UI_class:
    def __init__(self, master, search_path, frame_storing_path):
        self.search_path = search_path
        self.master = master
        self.frame_storing_path = frame_storing_path
        topframe = Frame(self.master)
        topframe.pack()

        self.query_img_frame = None
        self.hasPreviousQuery = False

        #Buttons
        topspace = Label(topframe).grid(row=0, columnspan=2)
        self.bbutton= Button(topframe, text=" Choose an video ", command=self.browse_query_img)
        self.bbutton.grid(row=1, column=1)
        self.cbutton = Button(topframe, text=" Estimate its venue ", command=self.show_venue_category)
        self.cbutton.grid(row=1, column=2)
        downspace = Label(topframe).grid(row=3, columnspan=4)

        self.venues = np.loadtxt("../venue-name.txt", dtype="str", delimiter="\n");
        self.categories = []
        for category in self.venues:
            self.categories.append(category.split("\t"))

        self.master.mainloop()


    def browse_query_img(self):

        if (self.hasPreviousQuery):
            self.query_img_frame.pack_forget()

        self.query_img_frame = Frame(self.master)
        self.query_img_frame.pack()
        from tkFileDialog import askopenfilename
        self.filename = tkFileDialog.askopenfile(title='Choose a Video File', initialdir=self.search_path).name

        allframes = os.listdir(self.frame_storing_path)
        self.videoname = self.filename.strip().split("/")[-1].replace(".mp4","")

        self.frames = []
        for frame in allframes:
            if self.videoname +"-frame" in frame:
                self.frames.append(self.frame_storing_path + frame)

        COLUMNS = len(self.frames)
        self.columns = COLUMNS
        image_count = 0
        display_limit = 5

        if COLUMNS == 0:
            self.frames.append("none.png")
            print("Please extract the key frames for the selected video first!!!")
            COLUMNS = 1

        for frame in self.frames:
            if (image_count >= display_limit):
                break

            r, c = divmod(image_count, COLUMNS)
            try:
                im = Image.open(frame)
                resized = im.resize((100, 100), Image.ANTIALIAS)
                tkimage = ImageTk.PhotoImage(resized)

                myvar = Label(self.query_img_frame, image=tkimage)
                myvar.image = tkimage
                myvar.grid(row=r, column=c)

                image_count += 1
                self.lastR = r
                self.lastC = c
            except Exception, e:
                continue

        self.query_img_frame.mainloop()


    def show_venue_category(self):
        if self.columns == 0:
            print("Please extract the key frames for the selected video first!!!")
        else:
            self.hasPreviousQuery = True
            self.filename = self.filename.split("/")[-1]
            # Please note that, you need to write your own classifier to estimate the venue category to show below.
            predictions = predictSVM(self.filename)
            predictions_strings = []
            classification_labels = ["1st:\n\n", "2nd:\n\n", "3rd:\n\n", "4th:\n\n", "5th:\n\n"]
            print(predictions)
            
            venue_text_primary = str(predictions[0]) # To change with result.

            for prediction in predictions:
                for category in self.categories:
                    if str(prediction) == category[0]:
                        predictions_strings.append(category[1])
                        break

            print(predictions_strings)

            os.chdir("UI")

            for i in range(5):
                venue_img = Image.open("venue_background.jpg")
                draw = ImageDraw.Draw(venue_img)

                font = ImageFont.truetype("arial.ttf",size=35)

                draw.text((25,25), (classification_labels[i] + predictions_strings[i]), (0, 0, 0), font=font)

                resized = venue_img.resize((100, 100), Image.ANTIALIAS)
                tkimage =ImageTk.PhotoImage(resized)

                myvar = Label(self.query_img_frame, image=tkimage)
                myvar.image= tkimage
                myvar.grid(row=self.lastR, column=self.lastC+i+1)

                #tkImage = draw.text = None

        self.query_img_frame.mainloop()


root = Tk()
window = UI_class(root,search_path='../presentation/', frame_storing_path='../presentation/frame/')
