# Importação das bibliotecas necessárias
import face_recognition
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Definição da classe Facerec
class Facerec:
    def __init__(self):
        # Inicialização das listas para armazenar codificações faciais e nomes conhecidos
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Fator de redimensionamento dos quadros de vídeo
        self.frame_resizing = 0.25
        
        # Criação do modelo TensorFlow para aprendizado de características faciais
        self.model = self.create_model()

    def create_model(self):
        # Criação de um modelo sequencial TensorFlow
        model = tf.keras.Sequential()
        
        # Adição de uma camada densa com ativação ReLU e entrada de forma (128,)
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(128,)))
        
        # Compilação do modelo com otimizador 'adam' e função de perda 'mse' (erro médio quadrático)
        model.compile(optimizer='adam', loss='mse')
        
        return model

    def load_encoding_images(self):
        # Carrega as codificações faciais e nomes conhecidos a partir de um arquivo CSV
        df = pd.read_csv('database.csv')
        self.known_face_names = df['Name'].tolist()
        self.known_face_encodings = [np.array(eval(encoding)) for encoding in df['Encoding']]

        # Se houver codificações faciais conhecidas, treina o modelo para reconstruí-las
        if len(self.known_face_names) > 0:
            X = np.array(self.known_face_encodings)
            y = np.array(self.known_face_names)
            self.model.fit(X, X, epochs=10)  # Treina o modelo para reconstruir as codificações existentes

    def save_encoding_image(self, name, encoding):
        # Salva uma nova codificação facial com o nome correspondente
        if name in self.known_face_names:
            return

        self.known_face_names.append(name)
        self.known_face_encodings.append(encoding.tolist())

        X = np.array(self.known_face_encodings)
        y = np.array(self.known_face_names)

        self.model.fit(X, X, epochs=10)  # Treina o modelo para reconstruir a nova codificação

        # Cria um DataFrame Pandas e salva as informações no arquivo CSV 'database.csv'
        df = pd.DataFrame({'Name': self.known_face_names, 'Encoding': self.known_face_encodings})
        df.to_csv('database.csv', index=False)

    def update_known_faces(self, frame, face_encoding):
        # Solicita ao usuário que insira o nome para uma face desconhecida e a salva
        name = self.ask_for_name()
        self.save_encoding_image(name, face_encoding)

    @staticmethod
    def ask_for_name():
        # Solicita ao usuário que insira seu nome para uma face desconhecida
        name = input("Unknown face detected. Please enter your name: ")
        return name

    def detect_known_faces(self, frame):
        # Redimensiona o quadro de vídeo e realiza o reconhecimento facial
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compara as codificações faciais com as codificações conhecidas
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(self.known_face_encodings, face_encoding))
                name = self.known_face_names[best_match_index]
            else:
                self.update_known_faces(frame, face_encoding)
                name = self.known_face_names[-1]

            face_names.append(name)

        # Converte as coordenadas das faces detectadas de volta ao tamanho original
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names

# Função principal
def main():
    # Cria uma instância da classe Facerec
    sfr = Facerec()
    
    # Carrega as codificações faciais e nomes conhecidos
    sfr.load_encoding_images()

    # Inicializa a captura de vídeo a partir da webcam (0)
    cap = cv2.VideoCapture(0)

    while True:
        # Lê um quadro de vídeo
        ret, frame = cap.read()

        # Detecta faces conhecidas no quadro de vídeo
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        # Desenha retângulos e nomes nas faces detectadas
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Mostra o quadro de vídeo com as informações das faces
        cv2.imshow("Frame", frame)

        # Verifica se a tecla 'Esc' (código 27) foi pressionada para encerrar o programa
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Libera a câmera e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Executa a função principal se o script for executado diretamente
if __name__ == "__main__":
    main()

