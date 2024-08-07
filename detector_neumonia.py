#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import ttk, font, filedialog, Entry
from keras import backend as K
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import tkinter as tk
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
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

class DetectorNeumonia:
    def __init__(self):
        self.array = None
        self.label = ""
        self.proba = 0.0
        self.heatmap = None
        self.model = self.load_model()

    def load_model(self):
        # Cargar modelo de detección de neumonía
        model = load_model('conv_MLP_84.h5')
        return model

    def preprocess(self, array):
        if len(array.shape) == 3 and array.shape[2] == 3:
            # Si la imagen tiene 3 canales, conviértela a escala de grises
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        array = cv2.resize(array, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array

    def grad_cam(self, array):
        # Preprocesar la imagen
        img = self.preprocess(array)
        img = tf.convert_to_tensor(img) 
    
        # Obtener el modelo y la última capa convolucional
        last_conv_layer = self.model.get_layer("conv10_thisone")
    
        with tf.GradientTape() as tape:
            # Habilitar el seguimiento de la imagen para calcular los gradientes
            tape.watch(img)
        
            # Realizar la predicción
            preds = self.model(img)
            argmax = tf.argmax(preds[0])  # Obtener el índice de la clase con mayor probabilidad
            output = preds[:, argmax]  # Salida para esa clase

        # Calcular los gradientes
        grads = tape.gradient(output, last_conv_layer.output)
    
        # Verificar si grads es None
        if grads is None:
            raise ValueError("Gradients are None, check the model and input.")

        # Promedio de los gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
        # Obtener los valores de la capa convolucional
        conv_layer_output_value = last_conv_layer.output[0] 

        # Multiplicar los valores de salida de la capa convolucional por los gradientes promediados
        for i in range(conv_layer_output_value.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads[i]
    
        # Crear el mapa de calor
        heatmap = tf.reduce_mean(conv_layer_output_value, axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU
        heatmap /= tf.reduce_max(heatmap)  # Normalizar

        # Convertir a numpy para usar con OpenCV
        heatmap = cv2.resize(heatmap.numpy(), (array.shape[1], array.shape[0]))  # Asegúrate de que coincide con la imagen
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
        # Superponer el mapa de calor en la imagen original
        img2 = cv2.resize(array, (array.shape[1], array.shape[0]))  # Asegúrate de que coincide con la imagen
        hif = 0.8
        superimposed_img = cv2.addWeighted(img2, 1 - hif, heatmap, hif, 0)  # Usar addWeighted para mezclar
        self.heatmap = superimposed_img

        return superimposed_img[:, :, ::-1]  # Cambiar el orden de los canales de color si es necesario

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

    def read_dicom_file(self, filepath):
        # Lectura de archivo DICOM
        ds = dicom.dcmread(filepath)
        array = ds.pixel_array
        img = Image.fromarray(array)
        return array, img    


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.detector = DetectorNeumonia()

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

        # NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        # RUN LOOP
        self.root.mainloop()

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
            self.detector.array, img2show = self.detector.read_dicom_file(filepath)
            img2show = Image.fromarray(self.detector.array)
            img2show = img2show.resize((250, 250), Image.BILINEAR)
            img2show = ImageTk.PhotoImage(img2show)
            self.text_img1.delete(1.0, tk.END)  # Limpiar cualquier texto previo
            self.text_img1.image_create(tk.END, image=img2show)
            self.button1["state"] = "enabled"  # Activar botón de predicción

    def run_model(self):
        self.detector.predict(self.detector.array)
        img_heatmap = Image.fromarray(self.detector.heatmap)
        img_heatmap = img_heatmap.resize((250, 250), Image.ANTIALIAS)
        img_heatmap = ImageTk.PhotoImage(img_heatmap)
        self.text_img2.delete(1.0, tk.END)  # Limpiar cualquier texto previo
        self.text_img2.image_create(tk.END, image=img_heatmap)
        self.text2.delete(1.0, tk.END)  # Limpiar cualquier texto previo
        self.text2.insert(tk.END, self.detector.label)
        self.text3.delete(1.0, tk.END)  # Limpiar cualquier texto previo
        self.text3.insert(tk.END, "{:.2f}".format(self.detector.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.detector.label, "{:.2f}".format(self.detector.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon="warning"
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, tk.END)
            self.text3.delete(1.0, tk.END)
            self.text_img1.delete(1.0, tk.END)
            self.text_img2.delete(1.0, tk.END)
            self.detector.array = None
            self.button1["state"] = "disabled"
            
if __name__ == "__main__":
    my_app = App()