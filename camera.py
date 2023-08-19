import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
from detection import AccidentDetectionModel

class InitialScreen:
    def __init__(self, window):
        self.window = window
        self.window.title("Tela Inicial")
        self.window.geometry("1000x600")
        self.window.configure(background="#f5f5f5")
        
        self.logo_image = Image.open("png.jpeg")  # Coloque o caminho para a imagem da logo
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        
        self.logo_label = tk.Label(window, image=self.logo_photo)
        self.logo_label.pack(pady=20)
        
        self.start_button = ttk.Button(window, text="Iniciar", command=self.start_program)
        self.start_button.pack()

    def start_program(self):
        self.window.destroy()  # Fecha a tela inicial
        root = tk.Tk()
        app = AccidentDetectionApp(root, "model.json", "model_weights.h5")
        root.mainloop()

class AccidentDetectionApp:
    def __init__(self, window, model_path, weights_path):
        self.window = window
        self.window.title("Accident Detection App")
        self.window.geometry("1000x600")
        self.window.configure(background="#f5f5f5")
        self.window.iconbitmap("icon.ico")  # Coloque o caminho para o ícone da janela

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = AccidentDetectionModel(model_path, weights_path)

        self.vid = cv2.VideoCapture('Teste\A2TH (1).gif')
        self.video_frame = tk.Label(window, relief="sunken")
        self.video_frame.pack(padx=20, pady=20, side=tk.LEFT)

        self.buttons_frame = ttk.Frame(window)
        self.buttons_frame.pack(padx=20, pady=20, side=tk.RIGHT)

        self.review_button = ttk.Button(self.buttons_frame, text="Revisar Frames", command=self.review_frames, style="Custom.TButton")
        self.review_button.pack(pady=10)

        self.help_button = ttk.Button(self.buttons_frame, text="Acionar Ajuda", command=self.acionar_ajuda, style="Custom.TButton")
        self.help_button.pack(pady=10)

        self.no_accident_button = ttk.Button(self.buttons_frame, text="Sem Acidente", style="Custom.TButton")
        self.no_accident_button.pack(pady=10)

        self.quit_button = ttk.Button(self.buttons_frame, text="Fechar", command=self.quit, style="Custom.TButton")
        self.quit_button.pack(pady=10)

        self.notified = False
        self.last_notification_time = 0

        # Inicialize o atributo frame_buffer
        self.frame_buffer = []
        self.buffer_size = 100  # Tamanho máximo do buffer (ajuste conforme necessário)

        self.current_frame_index = 0  # Índice do frame atual sendo exibido

        self.update()
        
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))
           
            self.frame_buffer.append(roi)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

            pred, prob = self.model.predict_accident(roi[np.newaxis, :, :])
            if pred == "Accident":
                prob = round(prob[0][0] * 100, 2)

                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, pred + " " + str(prob), (20, 30), self.font, 1, (255, 255, 0), 2)


            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_frame.configure(image=self.photo)
            self.video_frame.image = self.photo
            
        self.window.after(33, self.update)

    def review_frames(self):
        self.review_window = tk.Toplevel(self.window)
        self.review_window.title("Revisão de Frames")
        
        self.review_frame_index = 0
        self.review_frame_label = tk.Label(self.review_window, relief="sunken")
        self.review_frame_label.pack(padx=20, pady=20)

        # Crie a barra de deslizamento
        self.slider = ttk.Scale(self.review_window, from_=0, to=len(self.frame_buffer)-1, orient="horizontal", command=self.update_review_frame)
        self.slider.pack(pady=10)

        self.update_review_frame()

    def update_review_frame(self, event=None):
        self.review_frame_index = int(self.slider.get())
        if 0 <= self.review_frame_index < len(self.frame_buffer):
            frame = self.frame_buffer[self.review_frame_index]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_frame = pil_frame.resize((400, 400))

            review_photo = ImageTk.PhotoImage(image=pil_frame)
            self.review_frame_label.configure(image=review_photo)
            self.review_frame_label.image = review_photo

        review_photo = ImageTk.PhotoImage(image=pil_frame)
        review_frame_label = tk.Label(self.review_window, image=review_photo, relief="sunken")

    def show_previous_review_frame(self):
        if self.review_frame_index > 0:
            self.review_frame_index -= 1
            self.update_review_frame()

    def show_next_review_frame(self):
        if self.review_frame_index < len(self.frame_buffer) - 1:
            self.review_frame_index += 1
            self.update_review_frame()


    def acionar_ajuda(self):
        messagebox.showinfo("Ajuda Acionada", "Ajuda foi acionada!")

    def quit(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    initial_root = tk.Tk()
    initial_screen = InitialScreen(initial_root)
    initial_root.mainloop()
