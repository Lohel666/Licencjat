import work_on_trained_AI as AI
from tkinter import *
# C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/00000.ppm


def decode_road_sign():
    if(E1.get()):
        sign = AI.recognise_road_sign(E1.get())
        recognition_text_box.set(sign[0])
    else:
        recognition_text_box.set('empty path')

    label = Message(root, textvariable=recognition_text_box, relief=RAISED)


# 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/00000.ppm'
root = Tk()
root.title("AI - road signs")
root.geometry("300x150")
recognition_text_box = StringVar()
recognition_text_box.set("empty")
label = Message(root, textvariable=recognition_text_box,
                relief=RAISED, pady=5, padx=5, width=300)
B = Button(root, text="identify ", command=decode_road_sign)
B.place(x=50, y=95)
L1 = Label(root, text="Path")
L1.place(x=50, y=50)
E1 = Entry(root, bd=5)
E1.place(x=100, y=50)
label.pack()
root.mainloop()
