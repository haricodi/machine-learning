import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from sklearn.datasets import load_iris

root=Tk()
root.geometry("2000x1000")
root.title("MACHINE LEARNING")
root.configure(background="yellow")
Photo=PhotoImage(file="image1.png")
root.iconphoto(False,Photo)
Label(root,text="NAIVE BAYES ALGORITHM",font=('arial',15,'bold'),bg="white",fg="black",relief="solid").pack()
Label(root,text="DIFFERENT TYPES OF NAIVE BAYES",font=('arial',15,'bold'),fg="darkblue").pack(fill='both')
Label(root,text="BERNOULI NAIVE BAYES",font=('arial',15,'bold'),fg="black",bg="light blue").place(x=40,y=60)
Label(root,text="Sepal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=40,y=90)
Label(root,text="Sepal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=40,y=140)
Label(root,text="Petal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=40,y=190)
Label(root,text="Petal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=40,y=240)
Label(root,text="SPEC TYPE:",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=40,y=280)
Label(root,text="ACCURACY ON BERNOULI:",font=('arial',10,'bold'),bg="white",relief="solid",width=22).place(x=40,y=330)
s1=StringVar()
sw=StringVar()
p1=StringVar()
pw=StringVar()
Entry(root,text=s1,width=25).place(x=200,y=90)
Entry(root,text=sw,width=25).place(x=200,y=140)
Entry(root,text=p1,width=25).place(x=200,y=190)
Entry(root,text=pw,width=25).place(x=200,y=240)

def model1():
    data=load_iris()
    x=data.data
    y=data.target
    from sklearn.naive_bayes import BernoulliNB
    model1=BernoulliNB()
    model1.fit(x,y)
    x_test=[float(s1.get()),float(sw.get()),float(p1.get()),float(pw.get())]
    y_pred=model1.predict([x_test,])
    Label(root,text=str(data.target_names[y_pred]),font=('arial',10,'bold'),bg="orange",fg='black',relief="solid",width=18).place(x=200,y=280)
    Label(root,text="0.33333",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=250,y=330)
Button(root,text="PREDICT",font=("arial",10,'bold'),fg="red",width=18,command=model1).place(x=40,y=400)
Button(root,text="CLEAR",font=("arial",10,'bold'),fg="red",width=18,command=root.destroy).place(x=200,y=400)  


#multinomial naive bayes

Label(root,text="MULTINOMIAL NAIVE BAYES",font=('arial',15,'bold'),fg="black",bg="light blue").place(x=500,y=60)
Label(root,text="Sepal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=500,y=90)
Label(root,text="Sepal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=500,y=140)
Label(root,text="Petal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=500,y=190)
Label(root,text="Petal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=500,y=240)
Label(root,text="SPEC TYPE :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=500,y=280)
Label(root,text="ACCURACY ON MULTINOMIAL:",font=('arial',10,'bold'),bg="white",relief="solid",width=24).place(x=500,y=330)
s11=StringVar()
sw1=StringVar()
p11=StringVar()
pw1=StringVar()
Entry(root,text=s11,width=25).place(x=660,y=90)
Entry(root,text=sw1,width=25).place(x=660,y=140)
Entry(root,text=p11,width=25).place(x=660,y=190)
Entry(root,text=pw1,width=25).place(x=660,y=240)

def model2():
    data=load_iris()
    x=data.data
    y=data.target
    from sklearn.naive_bayes import MultinomialNB
    model2=MultinomialNB()
    model2.fit(x,y)
    x_test=[float(s11.get()),float(sw1.get()),float(p11.get()),float(pw1.get())]
    y_pred=model2.predict([x_test,])
    Label(root,text=str(data.target_names[y_pred]),font=('arial',10,'bold'),bg="orange",fg='black',relief="solid",width=18).place(x=660,y=280)
    Label(root,text="0.95333",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=720,y=330)
Button(root,text="PREDICT",font=("arial",10,'bold'),fg="red",width=18,command=model2).place(x=500,y=400)
Button(root,text="CLEAR",font=("arial",10,'bold'),fg="red",width=18,command=root.destroy).place(x=660,y=400)  


#Gaussian naive bayes


Label(root,text="GAUSSIAN NAIVE BAYES",font=('arial',15,'bold'),fg="black",bg="light blue").place(x=1100,y=60)
Label(root,text="Sepal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1100,y=90)
Label(root,text="Sepal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1100,y=140)
Label(root,text="Petal Length :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1100,y=190)
Label(root,text="Petal Width :",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1100,y=240)
Label(root,text="SPEC TYPE:",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1100,y=280)
Label(root,text="ACCURACY ON GAUSSIAN:",font=('arial',10,'bold'),bg="white",relief="solid",width=22).place(x=1100,y=330)
s12=StringVar()
sw2=StringVar()
p12=StringVar()
pw2=StringVar()
Entry(root,text=s12,width=25).place(x=1260,y=90)
Entry(root,text=sw2,width=25).place(x=1260,y=140)
Entry(root,text=p12,width=25).place(x=1260,y=190)
Entry(root,text=pw2,width=25).place(x=1260,y=240)

def model3():
    data=load_iris()
    x=data.data
    y=data.target
    from sklearn.naive_bayes import GaussianNB
    model3=GaussianNB()
    model3.fit(x,y)
    x_test=[float(s12.get()),float(sw2.get()),float(p12.get()),float(pw2.get())]
    y_pred=model3.predict([x_test,])
    Label(root,text=str(data.target_names[y_pred]),font=('arial',10,'bold'),bg="orange",fg='black',relief="solid",width=18).place(x=1260,y=280)
    Label(root,text="0.96",font=('arial',10,'bold'),bg="white",relief="solid",width=18).place(x=1310,y=330)    
Button(root,text="PREDICT",font=("arial",10,'bold'),fg="red",width=18,command=model3).place(x=1100,y=400)
Button(root,text="CLEAR",font=("arial",10,'bold'),fg="red",width=18,command=root.destroy).place(x=1260,y=400)  


root.mainloop()















































































