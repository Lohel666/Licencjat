from tkinter import *
from PIL import Image, ImageTk
import work_on_trained_AI as AI
from tkinter.filedialog import askopenfilename


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("AI - road signs")
        self.pack(fill=BOTH, expand=1)

        width = self.master.winfo_width()+400
        height = self.master.winfo_height()+250
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.master.resizable(False, False)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label='Open image', command=self.load_image)
        file.add_command(label='Exit', command=self.client_exit)
        menu.add_cascade(label='File', menu=file)
        text_box.set("Let's start!")

        B1 = Button(root, text="Select image ", command=self.load_image)
        B1.place(x=160, y=50)

        L1 = Label(root, text="Path")
        L1.place(x=10, y=95)

        self.E1 = Entry(root, bd=5, width=47)
        self.E1.place(x=40, y=95)

        B2 = Button(root, text="Identify ", command=self.decode_road_sign)
        B2.place(x=340, y=95)

    def load_image(self):
        path = askopenfilename()
        sign = AI.recognise_road_sign(path)
        text_box.set(sign[0])
        Message(root, textvariable=text_box, relief=RAISED)
        self.show_image(path)

    def show_image(self, path):
        load = Image.open(path).resize((64, 64))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=164, y=132)

    def client_exit(self):
        exit()

    def decode_road_sign(self):
        try:
            if(self.E1.get()):
                path = self.E1.get()
                sign = AI.recognise_road_sign(path)
                text_box.set(sign[0])
                self.show_image(path)
            else:
                text_box.set('ERR: decode_road_sign')
        except:
            text_box.set('File does not exist / wrong path')
        Message(root, textvariable=text_box, relief=RAISED)


root = Tk()
text_box = StringVar()
Message(root, textvariable=text_box,
        relief=RAISED, pady=5, padx=5, width=300).pack()


app = Window(root)
root.mainloop()
