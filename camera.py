import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from detection import AccidentDetectionModel
import tkinter

class InitialScreen:
    def __init__(self, window):
        self.window = window
        self.window.title("Accident Detection App")
        self.window.geometry("1920x1080")
        self.window.configure(background="#f07d2b")  # Fundo degradê
        
        self.logo_image = Image.open("png.png")  # Coloque o caminho para a imagem da logo
        self.logo_photo = ctk.CTkImage(self.logo_image, size=(640, 480))
        
        self.logo_label = ctk.CTkLabel(window, image=self.logo_photo, text='')
        self.logo_label.pack(pady=20)
        
        self.start_button = ctk.CTkButton(window, text="Iniciar", command=self.start_program)
        self.start_button.pack()

    def start_program(self):
        self.window.destroy()  # Fecha a tela inicial
        root = ctk.CTk()
        app = AccidentDetectionApp(root, "model.json", "model_weights.h5")
        root.mainloop()

class AccidentDetectionApp:
    def __init__(self, window, model_path, weights_path):
        self.window = window
        self.window.title("Accident Detection App")
        self.window.geometry("1920x1080")  # Aumentei a largura para acomodar os vídeos
        self.window.configure(background="#f5f5f5")

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = AccidentDetectionModel(model_path, weights_path)

        self.last_accident_time = 0

        self.buttons_frame = ctk.CTkFrame(window)
        self.buttons_frame.pack(padx=20, pady=20, side=ctk.RIGHT)

        self.buttons_help = ctk.CTkFrame(window)
        self.buttons_help.pack(padx=20, pady=20, side=ctk.RIGHT)

        self.review_button1 = ctk.CTkButton(self.buttons_frame, text="Revisar Imagens - Câmera 1", command=lambda idx=0: self.review_frames(idx))
        self.review_button1.pack(pady=10)

        self.review_button2 = ctk.CTkButton(self.buttons_frame, text="Revisar Imagens - Câmera 2", command=lambda idx=1: self.review_frames(idx))
        self.review_button2.pack(pady=10)

        self.review_button3 = ctk.CTkButton(self.buttons_frame, text="Revisar Imagens - Câmera 3", command=lambda idx=2: self.review_frames(idx))
        self.review_button3.pack(pady=10)

        self.review_button4 = ctk.CTkButton(self.buttons_frame, text="Revisar Imagens - Câmera 4", command=lambda idx=3: self.review_frames(idx))
        self.review_button4.pack(pady=10)

        self.help_button1 = ctk.CTkButton(self.buttons_help, text="Acionar ajuda - Câmera 1", command=lambda idx=0: self.acionar_ajuda(idx))
        self.help_button1.pack(pady=10)

        self.help_button2 = ctk.CTkButton(self.buttons_help, text="Acionar ajuda - Câmera 2", command=lambda idx=1: self.acionar_ajuda(idx))
        self.help_button2.pack(pady=10)

        self.help_button3 = ctk.CTkButton(self.buttons_help, text="Acionar ajuda - Câmera 3", command=lambda idx=2: self.acionar_ajuda(idx))
        self.help_button3.pack(pady=10)

        self.help_button4 = ctk.CTkButton(self.buttons_help, text="Acionar ajuda - Câmera 4", command=lambda idx=3: self.acionar_ajuda(idx))
        self.help_button4.pack(pady=10)

        self.no_accident_button = ctk.CTkButton(self.buttons_frame, text="Sem Acidente")
        self.no_accident_button.pack(pady=10)

        self.quit_button = ctk.CTkButton(self.buttons_frame, text="Fechar", command=self.quit)
        self.quit_button.pack(pady=10)

        self.notified = False
        self.last_notification_time = 0
        self.frame_buffers = [[] for _ in range(4)]
        self.buffer_size = 100

        self.current_frame_index = 0

        self.vid1 = cv2.VideoCapture('Teste\Vídeo sem título.mp4')
        self.vid2 = cv2.VideoCapture('Teste\Demo.gif')
        self.vid3 = cv2.VideoCapture('Teste\gif-acidente-1.gif')
        self.vid4 = cv2.VideoCapture('Teste\Demo.gif')

        self.video_frames = [
            ctk.CTkLabel(window, text=''),
            ctk.CTkLabel(window, text=''),
            ctk.CTkLabel(window, text=''),
            ctk.CTkLabel(window, text='')
        ]

        for frame in self.video_frames:
            frame.pack(padx=20, pady=20, side=ctk.LEFT)



        self.update()

    def update(self):
        ret1, frame1 = self.vid1.read()
        ret2, frame2 = self.vid2.read()
        ret3, frame3 = self.vid3.read()
        ret4, frame4 = self.vid4.read()

        if ret1 and ret2 and ret3 and ret4:
            # Adicione os quadros aos buffers de quadros correspondentes
            self.frame_buffers[0].append(frame1)
            self.frame_buffers[1].append(frame2)
            self.frame_buffers[2].append(frame3)
            self.frame_buffers[3].append(frame4)

            # Mantenha o tamanho do buffer limitado, removendo quadros mais antigos, se necessário
            buffer_size = 100
            for i in range(4):
                if len(self.frame_buffers[i]) > buffer_size:
                    self.frame_buffers[i].pop(0)

            
            self.detect_and_update(frame1, self.video_frames[0])
            self.detect_and_update(frame2, self.video_frames[1])
            self.detect_and_update(frame3, self.video_frames[2])
            self.detect_and_update(frame4, self.video_frames[3])


        if ret1 and ret2 and ret3 and ret4:
            resized_frame1 = cv2.resize(frame1, (250, 250))
            resized_frame2 = cv2.resize(frame2, (250, 250))
            resized_frame3 = cv2.resize(frame3, (250, 250))
            resized_frame4 = cv2.resize(frame4, (250, 250))

            self.update_video_frame(self.video_frames[0], resized_frame1)
            self.update_video_frame(self.video_frames[1], resized_frame2)
            self.update_video_frame(self.video_frames[2], resized_frame3)
            self.update_video_frame(self.video_frames[3], resized_frame4)

            # Realize a detecção e atualização para cada quadro
            self.detect_and_update(frame1, self.video_frames[0])
            self.detect_and_update(frame2, self.video_frames[1])
            self.detect_and_update(frame3, self.video_frames[2])
            self.detect_and_update(frame4, self.video_frames[3])


        self.window.after(33, self.update)


    def detect_and_update(self, frame, label):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = self.model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob), (20, 30), self.font, 1, (255, 255, 0), 2)

    def update_video_frame(self, label, frame):
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.configure(image=photo)
        label.image = photo
        self.detect_and_update(frame, label)


    def review_frames(self, camera_index):
        self.review_window = ctk.CTkToplevel(self.window)
        self.review_window.title(f"Revisão de Frames - Câmera {camera_index + 1}")

        self.review_frame_index = 0
        self.review_frame_label = ctk.CTkLabel(self.review_window, text='')
        self.review_frame_label.pack(padx=20, pady=20)

        self.slider = ctk.CTkSlider(self.review_window, from_=0, to=len(self.frame_buffers[camera_index]) - 1, orientation="horizontal", command=lambda event=None, idx=camera_index: self.update_review_frame(idx, event))
        self.slider.pack(pady=10)

        self.update_review_frame(camera_index)  # Chame a função de atualização com o índice da câmera

    def update_review_frame(self, camera_index, event=None):
        self.review_frame_index = int(self.slider.get())
        if 0 <= self.review_frame_index < len(self.frame_buffers[camera_index]):
            frame = self.frame_buffers[camera_index][self.review_frame_index]  # Use o índice da câmera adequado
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_frame = pil_frame.resize((800, 600))

            review_photo = ImageTk.PhotoImage(image=pil_frame)
            self.review_frame_label.configure(image=review_photo)
            self.review_frame_label.image = review_photo


    def show_previous_review_frame(self, idx):
        window_data = self.review_windows[idx]
        review_frame_index = window_data['frame_index']
        if review_frame_index > 0:
            window_data['frame_index'] -= 1
            self.update_review_frame(idx)

    def show_next_review_frame(self, idx):
        window_data = self.review_windows[idx]
        review_frame_index = window_data['frame_index']
        if review_frame_index < len(self.frame_buffer[idx]) - 1:
            window_data['frame_index'] += 1
            self.update_review_frame(idx)

    def acionar_ajuda(self, camera_index):
        # Cria a janela de ajuda
        help_window = ctk.CTkToplevel(self.window)
        help_window.geometry("600x400")  # Aumentei a altura para acomodar os botões
        help_window.title(f"Ajuda - Câmera {camera_index+1}")

        # Rótulo com a pergunta
        question_label = ctk.CTkLabel(help_window, text="Qual ajuda você deseja chamar?")
        question_label.pack(pady=10)

        # Função para mostrar a mensagem de ajuda
        def mostrar_mensagem_ajuda(mensagem):
            # Limpa qualquer rótulo de mensagem anterior
            if hasattr(self, "mensagem_label"):
                self.mensagem_label.destroy()

            # Cria um novo rótulo para a mensagem de ajuda
            self.mensagem_label = ctk.CTkLabel(help_window, text=mensagem)
            self.mensagem_label.pack(pady=10)

        # Botões de opções de ajuda
        def opcao1():
            mensagem = f"Ajuda acionada para a localização da câmera {camera_index+1}"
            mostrar_mensagem_ajuda(mensagem)

        button1 = ctk.CTkButton(help_window, text="Bombeiros", command=opcao1)
        button1.pack()

        def opcao2():
            mensagem = f"Ajuda acionada para a localização da câmera {camera_index+1}"
            mostrar_mensagem_ajuda(mensagem)

        button2 = ctk.CTkButton(help_window, text="Policia Civil", command=opcao2)
        button2.pack()

        def opcao3():
            mensagem = f"Ajuda acionada para a localização da câmera {camera_index+1}"
            mostrar_mensagem_ajuda(mensagem)

        button3 = ctk.CTkButton(help_window, text="SAMU", command=opcao3)
        button3.pack()

    def acidente(self):
        for i in range(4):
            acidente_window = ctk.CTkToplevel(self.window)
            acidente_window.geometry("300x150")
            acidente_window.title(f"Acidente - Câmera {i+1}")
            acidente_window.attributes("-topmost", True)
            acidente_label= ctk.CTkLabel(acidente_window, text=f'POSSÍVEL ACIDENTE na Câmera {i+1}, POR FAVOR VERIFICAR')
            acidente_label.pack(padx=20, pady=20)

    def quit(self):
        self.vid1.release()
        self.vid2.release()
        self.vid3.release()
        self.vid4.release()
        self.window.destroy()

if __name__ == "__main__":
    initial_root = ctk.CTk()
    initial_screen = InitialScreen(initial_root)
    initial_root.mainloop()
