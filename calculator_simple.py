import numpy as np
from numpy import sin, arcsin, arctan, cos, arccos , tan
from matplotlib import pyplot as plt
from tkinter import *
import pandas as pd
from tkinter import filedialog
from sklearn.linear_model import LogisticRegression, LinearRegression
π = np.pi
e = np.e
E = 10


class Calculator():
    def __init__(self):
        self.window = Tk()
        self.window.title("Calculator")
        self.window.geometry("520x600")
        self.window.resizable(True, True);
        self.window.configure(background="snow")
        self.screen_text = StringVar()
        self.expression =""
        self.simple()
        self.window.mainloop()
    def simple(self):
        self.clear(self.window)
        self.screen1 = Entry(self.window,font=("comic",20,"bold"),width=25,borderwidth=1,textvariable=self.screen_text)
        self.buttom0 = Button(self.window,text="0",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(0)).grid(row=4,column=0,pady=5,padx=5)
        self.buttom1 = Button(self.window,text="1",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(1)).grid(row=3,column=0,pady=5,padx=5)
        self.buttom2 = Button(self.window,text="2",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(2)).grid(row=3,column=1,pady=5,padx=5)
        self.buttom3 = Button(self.window,text="3",bg="DeepSkyBlue2",height=4,width=2,command=lambda :self.click(3)).grid(row=3,column=2,pady=5,padx=5)
        self.buttom4 = Button(self.window,text="4",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(4)).grid(row=2,column=0,pady=5,padx=5)
        self.buttom5 = Button(self.window,text="5",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(5)).grid(row=2,column=1,pady=5,padx=5)
        self.buttom6 = Button(self.window,text="6",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(6)).grid(row=2,column=2,pady=5,padx=5)
        self.buttom7 = Button(self.window,text="7",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(7)).grid(row=1,column=0,pady=5,padx=5)
        self.buttom8 = Button(self.window,text="8",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(8)).grid(row=1,column=1,pady=5,padx=5)
        self.buttom9 = Button(self.window,text="9",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(9)).grid(row=1,column=2,pady=5,padx=5)
        self.buttomPoint = Button(self.window,text=".",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click(".")).grid(row=4,column=1,pady=5,padx=5)
        self.buttomPI = Button(self.window,text="π",bg="DeepSkyBlue2",height=4,width=3,command=lambda :self.click("π")).grid(row=4,column=2,pady=5,padx=5)
        self.buttomAC = Button(self.window,text="AC",bg="DarkOrange3",height=4,width=3,command=lambda :self.clear_input()).grid(row=1,column=4,pady=5,padx=5)
        self.buttomDEl = Button(self.window,text="DEL",bg="DarkOrange3",height=4,width=3,command=lambda:self.delete()).grid(row=1,column=3,pady=5,padx=5)
        self.buttomSum = Button(self.window,text="+",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("+")).grid(row=3,column=3,pady=5,padx=5)
        self.buttomRes = Button(self.window,text="-",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("-")).grid(row=3,column=4,pady=5,padx=5)
        self.buttomMul = Button(self.window,text="*",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("*")).grid(row=2,column=3,pady=5,padx=5)
        self.buttomDiv = Button(self.window,text="/",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("/")).grid(row=2,column=4,pady=5,padx=5)
        self.buttomAns = Button(self.window,text="=",bg="DarkOrange3",height=4,width=3,command=lambda:self.term()).grid(row=5,column=2,pady=5,padx=5)
        self.buttomPow = Button(self.window,text="^",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("**")).grid(row=4,column=3,pady=5,padx=5)
        self.buttomSqrt = Button(self.window,text="√",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("np.sqrt")).grid(row=4,column=4,pady=5,padx=5)
        self.buttomEuler = Button(self.window,text="e",bg="DeepSkyBlue2",height=4,width=3,command=lambda:self.click("e")).grid(row=5,column=0,pady=5,padx=5)
        self.buttomEXP = Button(self.window,text="E",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("*E**")).grid(row=5,column=1,pady=5,padx=5)
        self.buttom_Start_Parenthesis = Button(self.window,text="(",bg="DarkOrange3",height=4,width=3,command=lambda:self.click("(")).grid(row=5,column=3,pady=5,padx=5)
        self.buttom_End_Parenthesis= Button(self.window,text=")",bg="DarkOrange3",height=4,width=3,command=lambda:self.click(")")).grid(row=5,column=4,pady=5,padx=5)
        #buttomSine = Button(self.window,text="GRAPHICS",bg="DarkOrange3",height=2,width=6).grid(row=2,column=0,pady=10)
        self.button_adv = Button(self.window,text="Advanced Options",bg="DarkOrange3",height=3,width=15,command=lambda:self.advanced()).grid(row=2, column=5,pady=5,padx=5)
        self.button_cons = Button(self.window,text="Constants",bg="DarkOrange3",height=3,width=15,command=lambda:self.advanced()).grid(row=3, column=5, pady=5,padx=5)
        self.button_scien = Button(self.window,text="Scientific",bg="DarkOrange3",height=3,width=15,command=lambda:self.scientific()).grid(row=1, column=5, pady=5,padx=5)
        self.screen1.grid(row=0,column=0,columnspan=5,padx=8,pady=5)
       
        
       
        
        
    def clear_input(self):
        self.expression = ""
        self.screen_text.set(" ")

        
    def delete(self):

        self.expression = self.expression[:(len(self.expression)-1)]
        self.screen_text.set(self.expression)
        
    def click(self,button_press):
        self.expression = self.screen1.get()

        self.expression += str(button_press)
        self.screen_text.set(self.expression)
        
    def term(self):
        self.expression = self.screen1.get()
        try:
            Term = str(eval(self.expression))
            self.expression = Term
        except ZeroDivisionError:
            Term = "math ERROR"
        except:
            Term = "syntax ERROR"
        self.screen_text.set(Term) 
        
        
        
    def to_classes(self,df):
        dic ={}
        y= []
        for i,x in enumerate(set(df.iloc[:,-1])):
            dic[x] = i
        for x in df.iloc[:,-1]:
            y.append(dic[x])
        return np.array(y)
            
            
        
    def fill(self, df):
        X,y = [],[]
        for i in range(len(df.values)):
            a = df.loc[i].values.flatten().tolist()
            X.append(a[:len(a)-2])
            y.append(a[-1])
        return np.array(X), y
        
    def log_reg(self):
        file_path = filedialog.askopenfilename()
        df = pd.read_csv(file_path)
               
        X, y = self.fill(df)
        y = self.to_classes(df)
  
       
        clf = LogisticRegression(random_state=0).fit(X, y)
        p = ",".join(input().split())
        a = f"np.array([[{p}]])"
        pred = eval((a))
        print(clf.predict(pred))
        
        
        #print("Predicted --", clf.predict(input()))
    def lin_reg(self):
        file_path = filedialog.askopenfilename()
        df = pd.read_csv(file_path)
               
        X, y = self.fill(df)
        y = self.to_classes(df)
  
       
        clf = LinearRegression().fit(X, y)
        p = ",".join(input().split())
        a = f"np.array([[{p}]])"
        pred = eval((a))
        print(clf.predict(pred))
        
    def graph(self):
        x = np.linspace(-np.pi, np.pi, 201)
        
        plt.plot(x, np.sin(x))
        
        plt.xlabel('Angle [rad]')
        
        plt.ylabel('sin(x)')
        
        plt.axis('tight')
        
        plt.show()
                
    
    def stat(self):
        pass
    def circuits(self):
        #hago OHM, Divisores y equivalentes
        f="building this..."
    
    def physics(self):
        #movimientos
        p="building this..."
        
    
    def scientific(self):
        self.clear(self.window)
        self.simple()
        self.button_scien = Button(self.window,text="Simple",bg="DarkOrange3",height=3,width=15,command=lambda:self.simple()).grid(row=1, column=5, pady=5,padx=5)
        self.buttomSin = Button(self.window,text="SIN",bg="gray65",height=4,width=3,command=lambda :self.click("sin")).grid(row=6,column=0,pady=5,padx=5)
        
    def advanced(self):
        self.clear(self.window)
        #self.button_return = Button(self.window,image = pygame.image.load('/home/andres/Pictures/areoNumber.jpg'),command=self.simple()).grid(row=0,column=0,pady=5,padx=5)
        self.button_lin_reg = Button(self.window,text="Lineal Regression",bg="DeepSkyBlue2",height=3,width=15,command=lambda:self.lin_reg()).grid(row=1, column=1, pady=30,padx=100)
        self.button_log_reg = Button(self.window,text="Logistic Regression",bg="DeepSkyBlue2",height=3,width=15,command=lambda :self.log_reg()).grid(row=1,column=0,pady=30,padx=50)
        self.button_graphs = Button(self.window,text="Graphs",bg="DeepskyBlue2",height=3,width=15, command=lambda: self.graph()).grid(row=2,column=0, pady=30, padx=50)
        self.button_stat = Button(self.window,text="Stat",bg="DeepSkyBlue", height=3, width=15, command=lambda: self.stat()).grid(row=2,column=1,pady=30,padx=100)
        self.button_circuits = Button(self.window,text="Circuits",bg="DeepSkyBlue", height=3, width=15, command=lambda: self.circuits()).grid(row=3,column=0,pady=30,padx=50)
        self.button_physics = Button(self.window,text="Physics",bg="DeepSkyBlue", height=3, width=15, command=lambda: self.physics()).grid(row=3,column=1,pady=30,padx=100)
    
    def clear(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()
calculator = Calculator()
calculator.window.mainloop()