import os
import sounddevice as sd
import numpy as np
import whisper
import openai
import scipy.io.wavfile as wav
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai


# Configura tu clave API de gemini (reemplaza "TU_API_KEY")
GEMINI_API_KEY = "TU_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)


# Configura tu clave API de openAI (reemplaza "TU_API_KEY")
API_KEY = "TU_API_KEY"

openai.api_key = API_KEY

# Configuración de audio
DURACION = 5
FRECUENCIA_MUESTREO = 16000

# Cargar modelo Whisper una vez
modelo_whisper = whisper.load_model("tiny")


# Función para interactuar con Gemini
def chat_con_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")  # Usa el modelo gemini-2.0-pro-exp-02-05
        response = model.generate_content(prompt)   # Envía el prompt al modelo
        return response.text  # Devuelve la respuesta generada
    except Exception as e:
        return f"Error: {e}"


def grabar_audio(nombre_archivo: str = "grabacion.wav") -> str:
    """Graba audio y lo guarda como archivo WAV."""
    try:
        print("Grabando...")
        audio = sd.rec(int(DURACION * FRECUENCIA_MUESTREO), samplerate=FRECUENCIA_MUESTREO, channels=1, dtype=np.int16)
        sd.wait()
        print("Grabación terminada.")
        wav.write(nombre_archivo, FRECUENCIA_MUESTREO, audio)
        return nombre_archivo
    except Exception as e:
        print(f"Error al grabar audio: {e}")
        return ""

def transcribir_audio(nombre_archivo: str) -> str:
    """Transcribe el audio a texto usando Whisper."""
    try:
        print("Transcribiendo...")
        resultado = modelo_whisper.transcribe(nombre_archivo, language="es", fp16=False)
        return resultado["text"]
    except Exception as e:
        print(f"Error en transcripción: {e}")
        return ""


def hablar(texto: str):
    """Convierte texto a voz y lo reproduce."""
    try:
        tts = gTTS(texto, lang="es")
        archivo_audio = "respuesta.mp3"
        tts.save(archivo_audio)

        audio = AudioSegment.from_mp3(archivo_audio)
        audio = audio.speedup(playback_speed=1.5)
        play(audio)

        os.remove(archivo_audio)
    except Exception as e:
        print(f"Error en la síntesis de voz: {e}")



# Datos de los sensores para la demo
datos_sensores = {
    "temperatura": 24,  # en °C
    "humedad": 45,      # en %
    "calidad aire": 60, # en %
}

# Crear un prompt para el modelo
prompt = f"""
Aqui tienes los datos de unos sensores dispuestos en mi casa
Datos del ambiente:
- Temperatura: {datos_sensores['temperatura']}°C
- Humedad: {datos_sensores['humedad']}%
- Calidad del aire: {datos_sensores['calidad aire']}%

Genera un informe breve sobre la situación ambiental en casa, basado en la información los sensores que te he dado.
Proporciona recomendaciones para mejorar el confort
En menos de 100 palabras.
\n
"""


def main():
    archivo_audio = grabar_audio()
    if not archivo_audio:
        return
    
    texto_transcrito = transcribir_audio(archivo_audio)
    print("\nTranscripción:", texto_transcrito)
    
    if texto_transcrito.strip():
        user_input = (texto_transcrito+prompt)
        respuesta = chat_con_gemini(user_input)
        print("IA:", respuesta)
        hablar(respuesta)
    else:
        print("No se detectó texto en la grabación.")



if __name__ == "__main__":
    main()

