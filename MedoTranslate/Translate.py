import whisper
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator


class Translate:
    def __init__(self):
        self.model = whisper.load_model('base')
        self.audio = None

    def transcribe(self, audio):
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)  # convert spectrogram to mel scale
        _, probs = self.model.detect_language(mel)
        srcLang = max(probs, key=probs.get)  # detect language
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)  # decode audio
        return (result.text, srcLang)

    def translateSpeech(self, srcText, srcLanguage, destLanguage):
        destText = GoogleTranslator(source=srcLanguage, target=destLanguage).translate(srcText)
        print(destText)
        tts = gTTS(destText)
        tts.save('tmp/output.mp3')
        playsound('tmp/output.mp3')
        return destText
