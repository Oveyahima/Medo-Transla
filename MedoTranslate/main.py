import Translate
import whisper
import speech_recognition as sr


class main:
    def __init__(self):
        self.trans = Translate.Translate()
        self.r = sr.Recognizer()

    def record(self):
        print("Start Speaking")
        with sr.Microphone() as source:
            recording = self.r.listen(source, 10, 5)
        print("Stop...")
        with open('tmp/input.wav', 'wb') as f:
            f.write(recording.get_wav_data())

    def speakTranslate(self):
        audio = whisper.load_audio('tmp/input.wav')
        audio = whisper.pad_or_trim(audio)  # load and trim to 30s
        srcText, srcLanguage = self.trans.transcribe(audio)
        print(f"Identified source Language {srcLanguage}")
        destLanguage = input("Enter destination Language:")
        print(srcText)
        destText = self.trans.translateSpeech(srcText, srcLanguage, destLanguage)

    def run(self):
        self.record()
        self.speakTranslate()


if __name__ == '__main__':
    m = main()
    m.run()
