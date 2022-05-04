from tkinter import *
import GUI1
import GUI2
from Vue import DateTime

global data
global date
data="Ito"
def call_GUI1():
    win2 = Toplevel(root)
    GUI1.GUI1(win2)
    return

def call_GUI2():
    win2 = Toplevel(root)

    date=DateTime.show(win2,data)

    print(date)
    return
def showData():
    global date
    print("DATA ITO ",date)
# the first gui owns the root window
if __name__ == "__main__":
    root = Tk()
    root.title('Caller GUI')
    root.minsize(720, 600)
    button_1 = Button(root, text='Call GUI1', width='20', height='20', command=showData)
    button_1.pack()
    button_2 = Button(root, text='Call GUI2', width='20', height='20', command=call_GUI2)
    button_2.pack()
    root.mainloop()