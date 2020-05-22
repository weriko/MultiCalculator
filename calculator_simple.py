#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:57:09 2020

@author: andres
"""
import math, cmath, numpy as np
from matplotlib import pyplot as plt
from tkinter import *

window = Tk()
window.title("pokemon")
window.geometry("400x600")
window.resizable(False,False);
window.configure(background="snow")
screen_text = StringVar()
expression =""

π = math.pi
e = math.e
E = 10
def clear():
    global expression
    expression = ""
    screen_text.set(" ")
    
def delete():
    global expression
    expression = expression[:(len(expression)-1)]
    screen_text.set(expression)
    
def click(b):
    global expression
    expression += str(b)
    screen_text.set(expression)
    
def term():
    global expression
    try:
        t = str(eval(expression))
        expression = t
    except ZeroDivisionError:
        t = "math ERROR"
    except ValueError or SyntaxError or NameError:
        t = "syntax ERROR"
    screen_text.set(t) 
    
screen1 = Entry(window,font=("comic",20,"bold"),width=25,borderwidth=1,textvariable=screen_text)
buttom0 = Button(window,text="0",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(0)).grid(row=4,column=0,pady=10,padx=10)
buttom1 = Button(window,text="1",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(1)).grid(row=3,column=0,pady=10,padx=10)
buttom2 = Button(window,text="2",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(2)).grid(row=3,column=1,pady=10,padx=10)
buttom3 = Button(window,text="3",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(3)).grid(row=3,column=2,pady=10,padx=10)
buttom4 = Button(window,text="4",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(4)).grid(row=2,column=0,pady=10,padx=10)
buttom5 = Button(window,text="5",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(5)).grid(row=2,column=1,pady=10,padx=10)
buttom6 = Button(window,text="6",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(6)).grid(row=2,column=2,pady=10,padx=10)
buttom7 = Button(window,text="7",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(7)).grid(row=1,column=0,pady=10,padx=10)
buttom8 = Button(window,text="8",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(8)).grid(row=1,column=1,pady=10,padx=10)
buttom9 = Button(window,text="9",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(9)).grid(row=1,column=2,pady=10,padx=10)
buttomPoint = Button(window,text=".",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click(".")).grid(row=4,column=1,pady=10,padx=10)
buttomPI = Button(window,text="π",bg="DeepSkyBlue2",height=2,width=2,command=lambda :click("π")).grid(row=4,column=2,pady=10,padx=10)
buttomAC = Button(window,text="AC",bg="DarkOrange3",height=2,width=2,command=lambda :clear()).grid(row=1,column=4,pady=10,padx=10)
buttomDEl = Button(window,text="DEL",bg="DarkOrange3",height=2,width=2,command=lambda:delete()).grid(row=1,column=3,pady=10,padx=10)
buttomSum = Button(window,text="+",bg="DarkOrange3",height=2,width=2,command=lambda:click("+")).grid(row=3,column=3,pady=10,padx=10)
buttomRes = Button(window,text="-",bg="DarkOrange3",height=2,width=2,command=lambda:click("-")).grid(row=3,column=4,pady=10,padx=10)
buttomMul = Button(window,text="*",bg="DarkOrange3",height=2,width=2,command=lambda:click("*")).grid(row=2,column=3,pady=10,padx=10)
buttomDiv = Button(window,text="/",bg="DarkOrange3",height=2,width=2,command=lambda:click("/")).grid(row=2,column=4,pady=10,padx=10)
buttomAns = Button(window,text="=",bg="DarkOrange3",height=2,width=2,command=lambda:term()).grid(row=4,column=4,pady=10,padx=10)
buttomPow = Button(window,text="^",bg="DarkOrange3",height=2,width=2,command=lambda:click("**")).grid(row=5,column=0,pady=10,padx=10)
buttomSqrt = Button(window,text="√",bg="DarkOrange3",height=2,width=2,command=lambda:click("**1/")).grid(row=5,column=1,pady=10,padx=10)
buttomEuler = Button(window,text="e",bg="DeepSkyBlue2",height=2,width=2,command=lambda:click("e")).grid(row=5,column=2,pady=10,padx=10)
buttomEXP = Button(window,text="E",bg="DarkOrange3",height=2,width=2,command=lambda:click("*E**")).grid(row=4,column=3,pady=10,padx=10)
#buttomSine = Button(window,text="GRAPHICS",bg="DarkOrange3",height=2,width=6).grid(row=2,column=0,pady=10)

screen1.grid(row=0,column=0,columnspan=5,padx=20,pady=20)
window.mainloop()