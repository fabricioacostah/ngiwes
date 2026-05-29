import cv2
import mediapipe as mp
import numpy as np
import platform
import os
from PIL import Image, ImageDraw, ImageFont

# 1. Inicializar MediaPipe configurado para DOS MANOS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# IDs de las puntas de los dedos
tip_ids = [4, 8, 12, 16, 20]

# Diccionario de comunicación extendido con EMOJIS reales
vocabulario_bimanual = {
    (5, 5): "¡HOLA! 👋👋 / ¡BIENVENIDOS! ✨",
    (0, 0): "GRACIAS 🙏 (Muestras Respeto)",
    (5, 0): "POR FAVOR... 🥺",
    (0, 5): "LO SIENTO 🙇 / DISCULPA 😔",
    (2, 2): "PAZ Y AMOR ✌️✌️ / AMISTAD 🕊️",
    (1, 1): "SÍ 👍 / DE ACUERDO ✅",
    (1, 5): "NECESITO AYUDA 🚨 / SOCORRO 🆘",
    (5, 2): "¡FELICITACIONES! 🎉 / BUEN TRABAJO 👏"
}

vocabulario_unamanual = {
    5: "¡ADIÓS! 👋 / HASTA LUEGO",
    2: "TODO BIEN 👍 / OK 👌",
    0: "ESPERA ✋ / PAUSA ⏸️"
}

# 2. Configuración inteligente de fuentes según el Sistema Operativo (Windows o macOS)
sistema_operativo = platform.system()

try:
    if sistema_operativo == "Windows":
        fuente_texto = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 16)
        fuente_titulo = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 18)
        fuente_emoji = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", 16)
        fuente_emoji_grande = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", 24)
    elif sistema_operativo == "Darwin":  # macOS (Tu MacBook Air)
        # Rutas modernas optimizadas para macOS (Ventura/Sonoma/Sequoia)
        ruta_arial = "/System/Library/Fonts/Supplemental/Arial.ttf" if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf") else "/Library/Fonts/Arial.ttf"
        ruta_emoji = "/System/Library/Fonts/Apple Color Emoji.ttc"
        
        fuente_texto = ImageFont.truetype(ruta_arial, 15)
        fuente_titulo = ImageFont.truetype(ruta_arial, 18)
        fuente_emoji = ImageFont.truetype(ruta_emoji, 14)
        fuente_emoji_grande = ImageFont.truetype(ruta_emoji, 22)
    else:
        fuente_texto = fuente_titulo = fuente_emoji = fuente_emoji_grande = ImageFont.load_default()
except Exception as e:
    print(f"Aviso: Usando fuente por defecto ({e})")
    fuente_texto = fuente_titulo = fuente_emoji = fuente_emoji_grande = ImageFont.load_default()

# 3. Iniciar captura de la cámara
cap = cv2.VideoCapture(0)
print("Intérprete bimanual iniciado en VS Code. Presiona 'q' en la ventana del video para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape

    # --- PANEL VISUAL DEL DICCIONARIO ---
    cv2.rectangle(frame, (15, 15), (490, 335), (35, 25, 15), -1)  # Fondo oscuro
    cv2.rectangle(frame, (15, 15), (490, 335), (255, 128, 0), 3)  # Borde naranja

    # Procesamiento de imágenes con MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    dedos_izq = None
    dedos_der = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lado_anatomico = handedness.classification[0].label 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos_abiertos = []

            # Lógica del Pulgar
            if lado_anatomico == "Left":
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    dedos_abiertos.append(1)
                else: dedos_abiertos.append(0)
            else: 
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    dedos_abiertos.append(1)
                else: dedos_abiertos.append(0)

            # Lógica de los otros 4 dedos
            for id in range(1, 5):
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                    dedos_abiertos.append(1)
                else:
                    dedos_abiertos.append(0)

            total_dedos_mano = dedos_abiertos.count(1)

            if lado_anatomico == "Left":
                dedos_izq = total_dedos_mano
            else:
                dedos_der = total_dedos_mano

    # --- DETERMINAR TRADUCCIÓN ---
    color_banner = (40, 40, 40)
    
    if dedos_izq is not None and dedos_der is not None:
        mensaje_pantalla = vocabulario_bimanual.get((dedos_izq, dedos_der), f"Combinación ({dedos_izq} + {dedos_der})")
        color_text_rgb = (0, 255, 0)  # Verde
        color_banner = (15, 45, 15)
        status_manos = f"Mano Izq: {dedos_izq} | Mano Der: {dedos_der}"
    
    elif dedos_izq is not None or dedos_der is not None:
        dedos_activos = dedos_izq if dedos_izq is not None else dedos_der
        mano_activa = "Izquierda" if dedos_izq is not None else "Derecha"
        mensaje_pantalla = vocabulario_unamanual.get(dedos_activos, "Formando palabra...")
        color_text_rgb = (255, 255, 0)  # Amarillo
        status_manos = f"Mano {mano_activa} detectada ({dedos_activos} dedos)"
    
    else:
        mensaje_pantalla = "SISTEMA EN ESPERA... 🔍"
        color_text_rgb = (255, 0, 0)  # ¡CORREGIDO AQUÍ! (Antes decía color_texto_rgb)
        status_manos = "Escaneando entorno visual..."

    # Fondo de la barra inferior
    cv2.rectangle(frame, (0, alto - 90), (ancho, alto), color_banner, -1)

    # =======================================================================
    # CAPA DE RENDERIZADO CON PILLOW (PIL) PARA INTEGRAR EMOJIS
    # =======================================================================
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Dibujar textos del Diccionario Gestual
    draw.text((30, 25), "DICCIONARIO GESTUAL", fill=(255, 128, 0), font=fuente_titulo)
    draw.text((30, 62), "5 + 5 ded. -> ¡HOLA! 👋👋 / BIENVENIDOS ✨", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 94), "0 + 0 ded. -> GRACIAS 🙏", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 126), "5 + 0 ded. -> POR FAVOR 🥺", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 158), "0 + 5 ded. -> LO SIENTO 🙇 / DISCULPA 😔", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 190), "2 + 2 ded. -> PAZ Y AMOR ✌️✌️", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 222), "1 + 1 ded. -> SÍ 👍 / DE ACUERDO ✅", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 254), "1 + 5 ded. -> NECESITO AYUDA 🚨", fill=(255, 255, 255), font=fuente_emoji)
    draw.text((30, 292), "Solo 1 mano abierta (5) -> ¡ADIÓS! 👋", fill=(255, 255, 0), font=fuente_emoji)

    # Dibujar barra inferior y resultado de traducción
    draw.text((25, alto - 75), status_manos, fill=(200, 200, 200), font=fuente_texto)
    draw.text((25, alto - 45), f"TRADUCCION: {mensaje_pantalla}", fill=color_text_rgb, font=fuente_emoji_grande)

    # Regresar el formato a OpenCV (BGR)
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    # =======================================================================

    cv2.imshow("Traductor Inteligente de Expresiones Gestuales", frame)

    # Presiona 'q' dentro de la ventana de video para cerrar el programa de forma limpia
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()