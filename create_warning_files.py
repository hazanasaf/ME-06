
from gtts import gTTS
import os

def create_warnings():
    if os.path.exists('./warnings/'): 
        os.system('rm -rf warnings') # delete warnings folder with all its contents
    os.system('mkdir warnings') # create empty warnings folder
    
    with open('warnings.txt', 'r') as warnings:
        lines = warnings.read().split('\n')
        for l in lines:
            if len(l.strip()) > 0:
                speech = gTTS(l, lang='en', slow=False)
                speech.save(f'warnings/{l}.mp3')
    os.system('cp ding.mp3 ./warnings/ding.mp3')

if __name__ == '__main__':
    create_warnings()