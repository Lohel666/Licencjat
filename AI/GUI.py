import work_on_trained_AI as AI
from tkinter import *
from tkinter.filedialog import askopenfilename


def decode_road_sign():
    try:
        if(E1.get()):
            sign = AI.recognise_road_sign(E1.get())
            recognition_text_box.set(sign[0])
        else:
            recognition_text_box.set('Empty path')
    except:
        recognition_text_box.set('File does not exist / wrong path')
    Message(root, textvariable=recognition_text_box, relief=RAISED)


def chose_file():
    sign = AI.recognise_road_sign(askopenfilename())
    recognition_text_box.set(sign[0])
    Message(root, textvariable=recognition_text_box, relief=RAISED)


root = Tk()


def window(main):
    main.title("AI - road signs")
    main.update_idletasks()
    width = main.winfo_width()+200
    height = main.winfo_height()
    x = (main.winfo_screenwidth() // 2) - (width // 2)
    y = (main.winfo_screenheight() // 2) - (height // 2)
    main.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    main.resizable(False, False)


window(root)
recognition_text_box = StringVar()
recognition_text_box.set("Empty")

Message(root, textvariable=recognition_text_box,
        relief=RAISED, pady=5, padx=5, width=300).pack()

L1 = Label(root, text="Path")
L1.place(x=50, y=50)

E1 = Entry(root, bd=5, width=40)
E1.place(x=100, y=50)

B1 = Button(root, text="Select image ", command=chose_file)
B1.place(x=50, y=95)

B2 = Button(root, text="Identify ", command=decode_road_sign,
            bd=5, font='Arial 9 bold')
B2.place(x=175, y=125)

root.mainloop()
