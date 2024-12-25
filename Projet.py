import cv2
import numpy as np
import tkinter as tk
from tkinter.ttk import *
from PIL import Image, ImageTk
import random
import Functions as f

colorFilter = False
detectionVisage = False
jailFilter = False
backgroundFilter = False
moneyFilter = False

bIsFirstTime = True

speed = 7

face_cascade = cv2.CascadeClassifier('Fichiers/haarcascade_frontalface_alt.xml')
imageBandit = 0
alphaBandit = 0
imageJail = cv2.imread('Projet/prison.png')
imgHeight, imgWidth, imgDepth = [0, 0, 0]

bill_max = 25
y_position = [0 for y in range(bill_max)]
x_positions = [random.randint(0, 480) for b in range(bill_max)]

y_position_bank = [0 for y in range(bill_max)]
x_positions_bank = [random.randint(0, 480) for b in range(bill_max)]

lastId = 0


def grab_frame(cam):
    ret, color1 = cam.read()
    dim = (640, 480)
    frame = cv2.resize(color1, dim, interpolation=cv2.INTER_AREA)
    return frame


def remove_all_filter():
    global colorFilter, moneyFilter, jailFilter, backgroundFilter, detectionVisage
    colorFilter = moneyFilter = jailFilter = backgroundFilter = detectionVisage = False


def apply_all_filter():
    global colorFilter, moneyFilter, jailFilter, backgroundFilter, detectionVisage, alphaBandit, imageBandit
    colorFilter = moneyFilter = jailFilter = backgroundFilter = True
    imageBandit = cv2.imread('Projet/bandit.png')
    alphaBandit = cv2.imread('Projet/bandit-alpha.png')
    alphaBandit = alphaBandit.astype(float) / 255
    imageBandit = imageBandit.astype(float)
    detectionVisage = True


def toggle_color_filter():
    global colorFilter
    colorFilter = not colorFilter


def toggle_money_money_filter():
    global moneyFilter
    moneyFilter = not moneyFilter


def toggle_jail_filter():
    global jailFilter, backgroundFilter, moneyFilter
    backgroundFilter, moneyFilter = False, False
    jailFilter = not jailFilter


def toggle_bank_filter():
    global backgroundFilter
    backgroundFilter = not backgroundFilter


def toggle_head_bandit():
    toggle_face_detection(0)


def toggle_mask_bandit():
    toggle_face_detection(1)


def toggle_face_detection(id):
    global imageBandit, alphaBandit, imgHeight, imgWidth, imgDepth
    if id == 0:
        imageBandit = cv2.imread('Projet/bandit.png')
        alphaBandit = cv2.imread('Projet/bandit-alpha.png')
    elif id == 1:
        imageBandit = cv2.imread('Projet/bandit2.png')
        alphaBandit = cv2.imread('Projet/bandit2-alpha.png')

    imgHeight, imgWidth, imgDepth = imageBandit.shape
    alphaBandit = alphaBandit.astype(float) / 255
    imageBandit = imageBandit.astype(float)

    global detectionVisage, lastId, bIsFirstTime
    if lastId == id or bIsFirstTime:
        detectionVisage = not detectionVisage
        bIsFirstTime = False
        lastId = 0

    lastId = id


def update_frame():
    ret, frame = cam.read()
    frame = grab_frame(cam)

    if detectionVisage:
        frame = f.detection_visage(frame, face_cascade, imageBandit, alphaBandit)

    if backgroundFilter:
        frame = f.filter_background(frame, moneyFilter, x_positions_bank, y_position_bank, speed)

    if moneyFilter:
        frame = f.filter_money(frame, speed, x_positions, y_position)

    if jailFilter:
        global imageJail
        frame = f.filter_jail(frame, imageJail)

    if colorFilter:
        frame = f.color_filter(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    panel.img = img
    panel.config(image=img)
    panel.after(10, update_frame)


root = tk.Tk()
root.title("Image Anna & Ivan")
root.attributes('-fullscreen', True)
root.bind('<Escape>', lambda e: root.destroy())

cam = cv2.VideoCapture(0)

panel = tk.Label(root)
panel.grid(row=1, column=1, columnspan=6, padx=300)

style = Style()
style.configure('W.TButton', font=('calibri', 20, 'bold'), background='black', foreground='black')

bandit1_button = Button(root, text="Filtre Bandit", style='W.TButton', command=toggle_head_bandit)
bandit1_button.grid(row=2, column=1)

bandit2_button = Button(root, text="Filtre Masque", style='W.TButton', command=toggle_mask_bandit)
bandit2_button.grid(row=2, column=2)

jail_button = Button(root, text="Filtre Banque", style='W.TButton', command=toggle_bank_filter)
jail_button.grid(row=2, column=3)

jail_button = Button(root, text="Filtre Billets", style='W.TButton', command=toggle_money_money_filter)
jail_button.grid(row=2, column=4)

color_button = Button(root, text="Filtre Police", style='W.TButton', command=toggle_color_filter)
color_button.grid(row=2, column=5)

jail_button = Button(root, text="Filtre Prison", style='W.TButton', command=toggle_jail_filter)
jail_button.grid(row=2, column=6)

jail_button = Button(root, text="Enlever tout", style='W.TButton', command=remove_all_filter)
jail_button.grid(row=3, column=3)

jail_button = Button(root, text="Ajouter tout", style='W.TButton', command=apply_all_filter)
jail_button.grid(row=3, column=4)

update_frame()

root.mainloop()

cam.release()
cv2.destroyAllWindows()
