#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry
from keras import backend as K
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import cv2
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import pydicom as dicom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


class DetectorNeumonia:
    def __init__(self):
        self.array = None
        self.label = ""
        self.proba = 0.0
        self.heatmap = None
        self.model = self.load_model()

    def load_model(self):
        # Aquí cargarías tu modelo de detección de neumonía
        model = load_model('conv_MLP_84.h5')
        return model

    def preprocess(self, array):
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array
    
    def grad_cam(self, array):
        img = self.preprocess(array)
        preds = self.model.predict(img)
        argmax = np.argmax(preds[0])
        output = self.model.output[:, argmax]
        last_conv_layer = self.model.get_layer("conv10_thisone")
        grads = tf.keras.backend.gradients(output, last_conv_layer.output)[0]
        pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))
        iterate = tf.keras.backend.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate(img)
        for filters in range(64):
            conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # normalize
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img2 = cv2.resize(array, (512, 512))
        hif = 0.8
        transparency = heatmap * hif
        transparency = transparency.astype(np.uint8)
        superimposed_img = cv2.add(transparency, img2)
        superimposed_img = superimposed_img.astype(np.uint8)
        self.heatmap = superimposed_img[:, :, ::-1]

    def predict(self, array):
        batch_array_img = self.preprocess(array)
        prediction = np.argmax(self.model.predict(batch_array_img))
        self.proba = np.max(self.model.predict(batch_array_img)) * 100
        if prediction == 0:
            self.label = "bacteriana"
        elif prediction == 1:
            self.label = "normal"
        elif prediction == 2:
            self.label = "viral"
        self.grad_cam(array)

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        # BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        # TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = tk.StringVar()
        self.result = tk.StringVar()

        # TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        # TWO IMAGE INPUT BOXES
        self.text_img1 = tk.Text(self.root, width=31, height=15)
        self.text_img2 = tk.Text(self.root, width=31, height=15)
        self.text2 = tk.Text(self.root)
        self.text3 = tk.Text(self.root)

        # BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        # FOCUS ON PATIENT ID
        self.text1.focus_set()

        # INSTANCE OF DETECTOR
        self.detector = DetectorNeumonia()

        # NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        # RUN LOOP
        self.root.mainloop()

    # METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            self.detector.array, img2show = self.read_dicom_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.ANTIALIAS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(tk.END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        self.detector.predict(self.detector.array)
        self.img2 = Image.fromarray(self.detector.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(tk.END, image=self.img2)
        self.text2.insert(tk.END, self.detector.label)
        self.text3.insert(tk.END, "{:.2f}".format(self.detector.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = tk.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.detector.label, "{:.2f}".format(self.detector.proba) + "%"]
            )
            tk.showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tk.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        tk.showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = tk.askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=tk.WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(0, "end")

if __name__ == "__main__":
    my_app = App()
