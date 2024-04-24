from transformers import pipeline, AutoTokenizer
from src.data_module import OddyseyDataset, get_data
from allosaurus.app import read_recognizer
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = read_recognizer('uni2005')

#train_df = pd.read_csv("data/EmoSPeech_phase_1_train_codalab.csv")
#test_df = pd.read_csv("data/EmoSPeech_phase_1_test_codalab.csv")

data = get_data(return_df=True)
train_data = OddyseyDataset()
#val_data = OddyseyDataset(split='val',
#                        data = data
#                        )

phonemes = []
for i in tqdm(range(53385)): #range(len(test_df))):
    #output = pipe(item[0], chunk_length_s=10, stride_length_s=(4, 2))
    url = train_data[i][0]#'data/train_segments/'+test_df.iloc[i,0]
    #audio = AudioSegment.from_mp3(url)
    #audio.export("temp.wav", format="wav")

    output = model.recognize(url, 'ipa') #"temp.wav"
    phonemes.append(output)

import pandas as pd
data = pd.read_csv('data/EmoSPeech_phase_2_train.csv')
phonemes = []
for i in tqdm(range(len(data))):
    audio_path = 'data/train_segments/' + data.iloc[i,0]
    audio = AudioSegment.from_mp3(audio_path)
    audio.export("temp.wav", format="wav")
    output = model.recognize("temp.wav", 'ipa')
    phonemes.append(output)

pd.DataFrame({'phonemes':phonemes}).to_csv('EmoSPeech_phase_2_train_phonemes.csv', index=False)
