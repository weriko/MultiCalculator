import math, cmath, numpy as np
from matplotlib import pyplot as plt
from tkinter import *
import pandas as pd

from tkinter import filedialog

π = math.pi
e = math.e
E = 10

class Calculator():
    def __init__(self):
        self.window = Tk()
        self.window.title("pokemon")
        self.window.geometry("400x600")
        self.window.resizable(True, True);
        self.window.configure(background="snow")
        self.screen_text = StringVar()
        self.expression =""
        self.simple()
        self.window.mainloop()
    def simple(self):
        self.clear(self.window)
        self.screen1 = Entry(self.window,font=("comic",20,"bold"),width=25,borderwidth=1,textvariable=self.screen_text)
        self.buttom0 = Button(self.window,text="0",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(0)).grid(row=4,column=0,pady=10,padx=10)
        self.buttom1 = Button(self.window,text="1",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(1)).grid(row=3,column=0,pady=10,padx=10)
        self.buttom2 = Button(self.window,text="2",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(2)).grid(row=3,column=1,pady=10,padx=10)
        self.buttom3 = Button(self.window,text="3",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(3)).grid(row=3,column=2,pady=10,padx=10)
        self.buttom4 = Button(self.window,text="4",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(4)).grid(row=2,column=0,pady=10,padx=10)
        self.buttom5 = Button(self.window,text="5",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(5)).grid(row=2,column=1,pady=10,padx=10)
        self.buttom6 = Button(self.window,text="6",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(6)).grid(row=2,column=2,pady=10,padx=10)
        self.buttom7 = Button(self.window,text="7",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(7)).grid(row=1,column=0,pady=10,padx=10)
        self.buttom8 = Button(self.window,text="8",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(8)).grid(row=1,column=1,pady=10,padx=10)
        self.buttom9 = Button(self.window,text="9",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(9)).grid(row=1,column=2,pady=10,padx=10)
        self.buttomPoint = Button(self.window,text=".",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click(".")).grid(row=4,column=1,pady=10,padx=10)
        self.buttomPI = Button(self.window,text="π",bg="DeepSkyBlue2",height=2,width=2,command=lambda :self.click("π")).grid(row=4,column=2,pady=10,padx=10)
        self.buttomAC = Button(self.window,text="AC",bg="DarkOrange3",height=2,width=2,command=lambda :self.clear_input()).grid(row=1,column=4,pady=10,padx=10)
        self.buttomDEl = Button(self.window,text="DEL",bg="DarkOrange3",height=2,width=2,command=lambda:self.delete()).grid(row=1,column=3,pady=10,padx=10)
        self.buttomSum = Button(self.window,text="+",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("+")).grid(row=3,column=3,pady=10,padx=10)
        self.buttomRes = Button(self.window,text="-",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("-")).grid(row=3,column=4,pady=10,padx=10)
        self.buttomMul = Button(self.window,text="*",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("*")).grid(row=2,column=3,pady=10,padx=10)
        self.buttomDiv = Button(self.window,text="/",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("/")).grid(row=2,column=4,pady=10,padx=10)
        self.buttomAns = Button(self.window,text="=",bg="DarkOrange3",height=2,width=2,command=lambda:self.term()).grid(row=4,column=4,pady=10,padx=10)
        self.buttomPow = Button(self.window,text="^",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("**")).grid(row=5,column=0,pady=10,padx=10)
        self.buttomSqrt = Button(self.window,text="√",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("**1/")).grid(row=5,column=1,pady=10,padx=10)
        self.buttomEuler = Button(self.window,text="e",bg="DeepSkyBlue2",height=2,width=2,command=lambda:self.click("e")).grid(row=5,column=2,pady=10,padx=10)
        self.buttomEXP = Button(self.window,text="E",bg="DarkOrange3",height=2,width=2,command=lambda:self.click("*E**")).grid(row=4,column=3,pady=10,padx=10)
        #buttomSine = Button(self.window,text="GRAPHICS",bg="DarkOrange3",height=2,width=6).grid(row=2,column=0,pady=10)
        self.button_adv = Button(self.window,text="Advanced Options",bg="DarkOrange3",height=3,width=15,command=lambda:self.advanced()).grid(row=9,column=0,pady=10,padx=10)
        self.screen1.grid(row=0,column=0,columnspan=5,padx=20,pady=20)
       
        
       
        
        
    def clear_input(self):
        self.expression = ""
        self.screen_text.set(" ")

        
    def delete(self):

        self.expression = self.expression[:(len(self.expression)-1)]
        self.screen_text.set(self.expression)
        
    def click(self,b):

        self.expression += str(b)
        self.screen_text.set(self.expression)
        
    def term(self):
        try:
            t = str(eval(self.expression))
            self.expression = t
        except ZeroDivisionError:
            t = "math ERROR"
        except ValueError or SyntaxError or NameError:
            t = "syntax ERROR"
        self.screen_text.set(t) 
        
    def log_reg(self):
     
        file_path = filedialog.askopenfilename()
        
        
        
    def advanced(self):
        self.clear(self.window)
        self.button_log_reg = Button(self.window,text="Logistic Regression",bg="DeepSkyBlue2",height=3,width=15,command=lambda :self.log_reg()).grid(row=4,column=0,pady=10,padx=10)
    
    
    def clear(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()
calculator = Calculator()
calculator.window.mainloop()
