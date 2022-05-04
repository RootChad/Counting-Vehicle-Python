import tkinter as tk

def GUI2(Frame):
    label = tk.Label(Frame, text="Hello from %s" % __file__)
    label.pack(padx=20, pady=20)
    return

if __name__ == "__main__":
    root = tk.Tk()
    GUI2(root)
    root.mainloop()