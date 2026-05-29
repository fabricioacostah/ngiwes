import cv2
import numpy as np
import platform
import os
import threading
import logging
from PIL import Image, ImageDraw, ImageFont

# Importación dinámica estándar para ejecución local nativa
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Sistema de logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InterpreteGestualPro:
    """
    Sistema empresarial de traducción de lenguaje de señas.
    Renderiza imágenes PNG dinámicas y procesa audio asíncrono.
    """
    
    def __init__(self):
        self.sistema_os = platform.system()
        self.configurar_fuentes()
        
        # Inicializar IA de MediaPipe
        self.hands_ai = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.tip_ids = [4, 8, 12, 16, 20]
        
        # Control de hilos y estados
        self.ultimo_mensaje_audio = ""
        self.bloqueo_audio = threading.Lock()
        
        # --- BLOQUE DE DETECCIÓN DE ASSETS CON ALINEACIÓN CORRECTA ---
        import sys
        if hasattr(sys, '_MEIPASS'):
            self.assets_dir = os.path.join(sys._MEIPASS, "assets")
        else:
            self.assets_dir = "assets"
            if not os.path.exists(self.assets_dir):
                os.makedirs(self.assets_dir)
        # -------------------------------------------------------------

        # Diccionarios de Datos
        self.vocabulario_bimanual = {
            (5, 5): ("HOLA / BIENVENIDOS", "hola.png"),
            (0, 0): ("GRACIAS (Muestras Respeto)", "gracias.png"),
            (5, 0): ("POR FAVOR...", "por_favor.png"),
            (0, 5): ("LO SIENTO / DISCULPA", "disculpa.png"),
            (2, 2): ("PAZ Y AMOR / AMISTAD", "paz.png"),
            (1, 1): ("SI / DE ACUERDO", "si.png"),
            (1, 5): ("NECESITO AYUDA / SOCORRO", "ayuda.png"),
            (5, 2): ("FELICITACIONES / BUEN TRAB.", "felicitaciones.png")
        }
        
        self.vocabulario_unamanual = {
            5: ("ADIOS / HASTA LUEGO", "adios.png"),
            2: ("TODO BIEN / OK", "ok.png"),
            0: ("ESPERA / PAUSA", "espera.png")
        }
        
        # Precargar imágenes en memoria RAM para optimizar rendimiento
        self.imagenes_cargadas = {}
        self.precargar_assets_graficos()
        
        logging.info("Arquitectura Pro cargada de forma exitosa.")

    def configurar_fuentes(self):
        """Carga las tipografías del sistema de manera segura."""
        try:
            if self.sistema_os == "Windows":
                self.f_texto = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 15)
                self.f_titulo = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 18)
            elif self.sistema_os == "Darwin":  # macOS
                ruta_arial = "/System/Library/Fonts/Supplemental/Arial.ttf" if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf") else "/Library/Fonts/Arial.ttf"
                self.f_texto = ImageFont.truetype(ruta_arial, 15)
                self.f_titulo = ImageFont.truetype(ruta_arial, 18)
            else:
                self.f_texto = self.f_titulo = ImageFont.load_default()
        except Exception as e:
            logging.warning(f"Error al cargar fuentes: {e}. Usando fallback por defecto.")
            self.f_texto = self.f_titulo = ImageFont.load_default()

    def precargar_assets_graficos(self):
        """Busca y aloja en memoria RAM las imágenes PNG para evitar latencia."""
        archivos_requeridos = [
            "hola.png", "gracias.png", "por_favor.png", "disculpa.png", "paz.png", 
            "si.png", "ayuda.png", "felicitaciones.png", "adios.png", "ok.png", 
            "espera.png", "sistema_espera.png"
        ]
        
        for archivo in archivos_requeridos:
            ruta_completa = os.path.join(self.assets_dir, archivo)
            if os.path.exists(ruta_completa):
                try:
                    # Forzamos conversión a RGBA para asegurar la transparencia del PNG
                    img = Image.open(ruta_completa).convert("RGBA")
                    self.imagenes_cargadas[archivo] = img
                    logging.info(f"Imagen vinculada con éxito: {archivo}")
                except Exception as e:
                    logging.error(f"No se pudo procesar el archivo {archivo}: {e}")
            else:
                logging.warning(f"Asset faltante: '{ruta_completa}'. Se usará reemplazo digital dinámico.")

    def reproducir_audio(self, texto_frase: str):
        """Ejecuta la pista de audio en un hilo independiente para no congelar la cámara."""
        def tarea_voz():
            with self.bloqueo_audio:
                palabra = texto_frase.split("/")[0].strip()
                if self.sistema_os == "Darwin":
                    os.system(f"say -v Paulina '{palabra}' &")
                elif self.sistema_os == "Windows":
                    try:
                        import pyttsx3
                        eng = pyttsx3.init()
                        eng.say(palabra)
                        eng.runAndWait()
                    except ImportError:
                        pass
                        
        threading.Thread(target=tarea_voz, daemon=True).start()

    def contar_dedos(self, hand_landmarks, lado_anatomico: str) -> int:
        """Determina la cantidad de dedos levantados por mano."""
        dedos_abiertos = []
        # Pulgar
        if lado_anatomico == "Left":
            dedos_abiertos.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
        else:
            dedos_abiertos.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
        # Resto de los dedos
        for id in range(1, 5):
            if hand_landmarks.landmark[self.tip_ids[id]].y < hand_landmarks.landmark[self.tip_ids[id] - 2].y:
                dedos_abiertos.append(1)
            else:
                dedos_abiertos.append(0)
        return dedos_abiertos.count(1)

    def renderizar_ui(self, frame, mensaje: str, nombre_imagen: str, status: str, color_texto: tuple):
        """Construye la UI superponiendo gráficos PNG con transparencia perfecta."""
        alto, ancho, _ = frame.shape
        
        # Marcos Geométricos Avanzados (OpenCV)
        cv2.rectangle(frame, (15, 15), (450, 335), (35, 25, 15), -1)      # Panel Diccionario
        cv2.rectangle(frame, (15, 15), (450, 335), (255, 128, 0), 2)
        cv2.rectangle(frame, (465, 15), (625, 165), (20, 20, 40), -1)     # Panel de Imagen Gráfica
        cv2.rectangle(frame, (465, 15), (625, 165), (0, 255, 255), 2)
        
        color_fondo_inferior = (15, 45, 15) if color_texto == (0, 255, 0) else (40, 40, 40)
        cv2.rectangle(frame, (0, alto - 90), (ancho, alto), color_fondo_inferior, -1)

        # Capa de renderizado híbrido (Pillow)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Escribir textos limpios en el Diccionario
        draw.text((30, 25), "DICCIONARIO GESTUAL", fill=(255, 128, 0), font=self.f_titulo)
        comandos = [
            ("5 + 5 ded. -> HOLA / BIENVENIDOS", 62, (255, 255, 255)),
            ("0 + 0 ded. -> GRACIAS", 94, (255, 255, 255)),
            ("5 + 0 ded. -> POR FAVOR...", 126, (255, 255, 255)),
            ("0 + 5 ded. -> LO SIENTO / DISCULPA", 158, (255, 255, 255)),
            ("2 + 2 ded. -> PAZ Y AMOR", 190, (255, 255, 255)),
            ("1 + 1 ded. -> SI / DE ACUERDO", 222, (255, 255, 255)),
            ("1 + 5 ded. -> NECESITO AYUDA", 254, (255, 255, 255)),
            ("Solo 1 mano (5) -> ADIOS!", 292, (255, 255, 0))
        ]
        for texto, y_pos, color in comandos:
            draw.text((30, y_pos), texto, fill=color, font=self.f_texto)

        # --- MOTOR GRÁFICO: INCORPORACIÓN DE IMÁGENES PNG ---
        if nombre_imagen in self.imagenes_cargadas:
            img_asset = self.imagenes_cargadas[nombre_imagen]
            # Redimensionar dinámicamente para que quepa perfectamente centrado en el recuadro
            img_redimensionada = img_asset.resize((120, 120), Image.Resampling.LANCZOS)
            # Pegamos la imagen usando su propio canal alfa como máscara de transparencia
            frame_pil.paste(img_redimensionada, (485, 30), img_redimensionada)
        else:
            # Si el archivo físico no existe, genera una interfaz digital de reemplazo limpia
            draw.rectangle([(485, 30), (605, 150)], fill=(30, 30, 50))
            texto_fallback = nombre_imagen.replace(".png", "").upper()
            draw.text((495, 80), f"[ {texto_fallback} ]", fill=(0, 255, 255), font=self.f_texto)

        # Textos Inferiores de Traducción
        draw.text((25, alto - 75), status, fill=(200, 200, 200), font=self.f_texto)
        draw.text((25, alto - 45), f"TRADUCTION: {mensaje}", fill=color_texto, font=self.f_titulo)

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    def ejecutar(self):
        """Bucle de ejecución principal."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cámara web inaccesible.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_ai.process(image_rgb)

            dedos_izq, dedos_der = None, None

            # Procesar datos topológicos de MediaPipe
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    lado = handedness.classification[0].label 
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    total_dedos = self.contar_dedos(hand_landmarks, lado)
                    if lado == "Left": dedos_izq = total_dedos
                    else: dedos_der = total_dedos

            # Inferencia y selección de Imagen representativa
            if dedos_izq is not None and dedos_der is not None:
                mensaje, img_name = self.vocabulario_bimanual.get((dedos_izq, dedos_der), (f"Combinacion ({dedos_izq}+{dedos_der})", "sistema_espera.png"))
                color, status = (0, 255, 0), f"Mano Izq: {dedos_izq} | Mano Der: {dedos_der}"
            elif dedos_izq is not None or dedos_der is not None:
                activos = dedos_izq if dedos_izq is not None else dedos_der
                mano_str = "Izquierda" if dedos_izq is not None else "Derecha"
                mensaje, img_name = self.vocabulario_unamanual.get(activos, ("Formando palabra...", "sistema_espera.png"))
                color, status = (255, 255, 0), f"Mano {mano_str} detectada ({activos} dedos)"
            else:
                mensaje, img_name = "SISTEMA EN ESPERA...", "sistema_espera.png"
                color, status = (255, 0, 0), "Escaneando entorno visual..."

            # Disparador del controlador de audio asíncrono
            if mensaje != self.ultimo_mensaje_audio:
                self.ultimo_mensaje_audio = mensaje
                if mensaje not in ["SISTEMA EN ESPERA...", "Formando palabra..."] and not mensaje.startswith("Combinacion"):
                    self.reproducir_audio(mensaje)

            # Renderizar interfaz e imágenes
            frame_final = self.renderizar_ui(frame, mensaje, img_name, status, color)
            cv2.imshow("Traductor Pro de Lenguaje de Senas", frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands_ai.close()

if __name__ == '__main__':
    app = InterpreteGestualPro()
    app.ejecutar()