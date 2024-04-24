import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
import librosa
from tqdm import tqdm
import re
from sklearn.model_selection import StratifiedKFold
from funasr.utils.load_utils import load_audio_text_image_video

def get_transcripts(data):
    transcripts = []
    for i in tqdm(range(len(data))):
        if data['audio_path'][i].startswith('Augmented'):
            transcript_path =  '/home/jaime/datasets/Odyssey_SER/' + 'Transcripts/'+ data['audio_path'][i][10:-3]+'txt'
        else:
            transcript_path =  '/home/jaime/datasets/Odyssey_SER/' + 'Transcripts/'+ data['audio_path'][i][:-3]+'txt'
        t = open(transcript_path, "r")
        transcript = t.read()
        transcripts.append(transcript)

    output = pd.DataFrame({'audio_path': data['audio_path'], 'transcript': transcripts})
    return output

def split_data(data):
    partitions_txt = pd.read_csv('/home/jaime/datasets/Odyssey_SER/'+'Partitions.txt', sep=';', header=None, names=['partition', 'audio_path_txt'])
    partitions_txt['audio_path_txt'] = partitions_txt['audio_path_txt'].str.strip()
    df_merged = pd.merge(data, partitions_txt, how='left', left_on='audio_path', right_on='audio_path_txt')

    df_merged.drop('audio_path_txt', axis=1, inplace=True)
    train = df_merged[df_merged['partition'] == 'Train']
    valid = df_merged[df_merged['partition'] == 'Development']
    train = train.iloc[:,:-1]
    valid = valid.iloc[:,:-1]
    return train, valid

def get_data(return_df=False, augmented = False, path='/home/jaime/datasets/Odyssey_SER/'+'Labels/labels.txt'):
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    data = []
    data_dict = {'audio_path': None, 'emotion': None, 'valence_A': None, 'valence_V': None, 'valence_D': None, 'workers': {}}
    current_worker_id = None
    worker=False
    for i in tqdm(range(len(lines))):
        line = lines[i]
        parts = line.split(';')
        if not line.strip():
            if data_dict['audio_path']!=None:
                data.append(data_dict)
            data_dict = {'audio_path': None, 'emotion': None, 'valence_A': None, 'valence_V': None, 'valence_D': None, 'workers': {}}

        elif line.strip():
            if line.split(';')[0].startswith('MSP') and augmented:
                worker=False
            if (parts[1] == ' N' or parts[1] == ' H' or parts[1] == ' Neutral' or parts[1] == ' Happy') and augmented:
                worker=True
                continue
            elif not worker:
                if line.split(';')[0].startswith('WORKER'):
                    current_worker_id = parts[0]
                    main_emotion = parts[1]
                    secondary_emotion = parts[2]
                    valence_A = float(re.search(r'\d+\.\d+', line.split(';')[3])[0])
                    valence_V = float(re.search(r'\d+\.\d+', line.split(';')[4])[0])
                    valence_D = float(re.search(r'\d+\.\d+', line.split(';')[5])[0])
                elif line.split(';')[0].startswith('MSP'):
                    if not augmented:
                        audio_path = parts[0]
                    elif augmented:
                        audio_path = 'Augmented/'+parts[0]
                    emotion = parts[1]
                    valence_A = float(re.search(r'\d+\.\d+', line.split(';')[2])[0])
                    valence_V = float(re.search(r'\d+\.\d+', line.split(';')[3])[0])
                    valence_D = float(re.search(r'\d+\.\d+', line.split(';')[4])[0])

                if audio_path:
                    data_dict['audio_path'] = audio_path
                    data_dict['emotion'] = emotion
                    data_dict['valence_A'] = valence_A
                    data_dict['valence_V'] = valence_V
                    data_dict['valence_D'] = valence_D
                    audio_path = None
                if current_worker_id:
                    data_dict['workers'][current_worker_id] = {
                        'main_emotion': main_emotion,
                        'secondary_emotion': secondary_emotion,
                        'valence_A': valence_A,
                        'valence_V': valence_V,
                        'valence_D': valence_D
                    }
                    current_worker_id = None

    if return_df:
        combined_data = []
        emotions_list = ['Neutral', 'Sad', 'Angry', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt']
        for item in data:
            row = [item['audio_path'], item['emotion'].replace(' ', '')]
            # Main emotions
            worker_main_emotions = [worker_info['main_emotion'] for worker_info in item['workers'].values()]
            worker_main_emotions = [s.replace(' ', '') for s in worker_main_emotions]
            # Check for emotions with name having 'Other-'
            other_emotions = []
            for i in range(len(worker_main_emotions)):
                if '-' in worker_main_emotions[i]:
                    # maintain the second part of the emotion
                    other_emotions.append(worker_main_emotions[i].split('-')[1])
            if other_emotions==[]:
                other_emotions.append('None')
            else:
                other_emotions = [s.lower() for s in other_emotions]
            # Count the number of times each emotion appears
            num_emotions = {emotion: worker_main_emotions.count(emotion) for emotion in emotions_list}
            num_emotions = list(num_emotions.values())
            row.extend(num_emotions)
            row.append(other_emotions)

            # Secondary emotions
            worker_secondary_emotios = [worker_info['secondary_emotion'] for worker_info in item['workers'].values()]
            worker_secondary_emotios = [word.strip() for s in worker_secondary_emotios for word in s.split(',')]
            unique_list = []
            sencond_emotions = []
            for word in worker_secondary_emotios:
                if word in emotions_list:
                    sencond_emotions.append(word)
                elif word not in unique_list:
                    if '-' in word:
                        unique_list.append(word.split('-')[1])
                    else:
                        unique_list.append(word)
            if unique_list == []:
                unique_list.append('None')
            else:
                unique_list = [s.lower() for s in unique_list]
            num_emotions_sencond = {emotion: sencond_emotions.count(emotion) for emotion in emotions_list}
            num_emotions_sencond = list(num_emotions_sencond.values())
            row.extend(num_emotions_sencond)
            row.append(unique_list)
            combined_data.append(row)

        second_emotions_list = ['second_'+emotion for emotion in emotions_list]
        columns = ['audio_path', 'emotion'] + emotions_list + ['Other_emotions'] + second_emotions_list + ['Other_secondary_emotions']
        data = pd.DataFrame(combined_data)
        data.columns = columns
        # Gender
        gender = pd.read_csv('/home/jaime/datasets/Odyssey_SER/'+'Labels/labels_consensus.csv')
        gender.rename(columns={'FileName': 'audio_path'}, inplace=True)
        data = pd.merge(gender[['audio_path', 'Gender']], data, on='audio_path')
        return data
    else:
        return data

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")

def collate_fn_odyssey(batch):
    audio_paths, text_tokens, gender, targets = zip(*batch)
    audio_inputs = []
    for audio in audio_paths:
        y, sr = librosa.load(audio, sr=16000)

        inputs = feature_extractor(
            y, sampling_rate=sr, return_tensors="np"
        )

        input_features = torch.tensor(inputs['input_features'])
        audio_inputs.append(input_features)
    audio_inputs=torch.stack(audio_inputs).squeeze(1)
    text_tokens = [torch.tensor(text_token) for text_token in text_tokens]
    return [audio_inputs, text_tokens, torch.tensor(gender), torch.tensor(targets)]


def get_agree(data, agree):
        emotion_order = ['Angry','Sad','Happy','Surprise','Fear','Disgust','Contempt','Neutral']

        df=pd.DataFrame()
        for emotion in emotion_order:
            df[emotion]=data[emotion].values
        #x = pd.DataFrame(x).T
        x = df.div(df.sum(axis=1), axis=0)

        delete_rows = []
        for i in tqdm(range(len(x))):
            if (x.iloc[i,:].argmax()==7 or x.iloc[i,:].argmax()==2) and x.iloc[i,:].max()<agree:
                delete_rows.append(i)

        red_data = data.drop(delete_rows, axis=0)
        red_data = red_data.reset_index(drop=True)
        return red_data

class OddyseyDataset():
    def __init__(self, data, split = None, data_dir = '/home/jaime/datasets/Odyssey_SER/', **kwargs):
        self.data_dir = data_dir
        self.split = split
        self.data = data
        self.remove_none_other = kwargs.get('remove_none_other', True)
        self.agree = kwargs.get('agree', 0.0)
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 4)
        self.text = kwargs.get('transcript', True)
        self.phones = kwargs.get('phones', False)
        self.llm_name = kwargs.get('llm_name', 'gemma')

        if self.remove_none_other:
            self.data = self.remove_none_and_other(self.data)
        if self.split!=None:          
            self.train, self.val = split_data(self.data)
            self.train.reset_index(drop=True, inplace=True)
            self.val.reset_index(drop=True, inplace=True)
        if self.split == 'train':
            self.train = get_agree(self.train, self.agree)
            self.data = self.train
        elif self.split == 'val':
            self.data = self.val

        if self.llm_name=='phi':
            self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        elif self.llm_name=='gemma':
            self.gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

        self.transcripts = get_transcripts(self.data)

    def remove_none_and_other(self, data):
        data = data[data['emotion'] != 'X']
        data = data[data['emotion'] != 'O']
        data = data.reset_index(drop=True)
        return data

    def target_to_id(self, target):
        if self.remove_none_other:
            target_list=['N', 'S', 'D', 'H', 'F', 'A', 'C', 'U']
        else:
            target_list=['N', 'S', 'D', 'H', 'F', 'A', 'C', 'U', 'X', 'O']
        output=target_list.index(target)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data_dir + 'Audios/' + self.data['audio_path'][idx]
        target = self.data['emotion'][idx]
        target = self.target_to_id(target)
        gender = self.data['Gender'][idx]
        gender = 0 if gender == 'Female' else 1
        extra_prompt = ""
        if self.text:
            transcripts = self.transcripts['transcript'][idx]
            extra_prompt += f"The transcription is: {transcripts}\n"
            
        query=f"""Instruction: \nGiven the following audio information and the transcription, predict the emotion of the speaker.\nThe emotion can be one of the following: [ neutral, sad, angry, happy, surprise, fear, disgust, contempt]\n{extra_prompt}Audio information:\n\nOutput:\nemotion ="""

        if self.llm_name=='phi':
            text_tokens = self.phi_tokenizer(query)['input_ids']
        elif self.llm_name=='gemma':
            text_tokens = self.gemma_tokenizer(query)['input_ids']
        return audio_path, text_tokens, gender, target


class OdysseyDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.batch_size = kwargs.get('batch_size', 8)
        self.num_workers = kwargs.get('num_workers', 4)
        self.shuffle = kwargs.get('shuffle', False)

    def setup(self, stage=None):
        data = get_data(return_df=True)
        self.train_data = OddyseyDataset(split='train',
                                        data = data,
                                        **self.config
                                        )
        self.val_data = OddyseyDataset(split='val',
                                    data = data, 
                                    **self.config
                                    )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_odyssey
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_odyssey
        )


class EmoSPeechDataset():
    def __init__(self, data, **kwargs):
        self.data = data
        self.model = kwargs.get('model', 'WhisperGemma')
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        self.gender = kwargs.get('gender', False)
        self.phones = kwargs.get('phones', False)
        self.audio_path = kwargs.get('audio_path', False)
        self.predict = kwargs.get('predict', False)
        self.audio_encoder_name = kwargs.get('audio_encoder_name', 'emotion2vec')
        if self.audio_encoder_name=='emotion2vec':
            self.audio_tokenizer = torch.load("weights/emotion2vec_tokenizer.ckpt")

    def target_to_id(self, target):
        target_list=['neutral', 'disgust', 'anger', 'joy', 'sadness', 'fear']
        output=target_list.index(target)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = 'data/'+self.data['path'][idx]
        transcripts = self.data['transcription'][idx]
        if not self.predict:
            target = self.data['label'][idx]
            target = self.target_to_id(target)

        if self.model=='QwenAudio':
            prompt = '<|startofanalysis|><|es|><|keyword|><|en|><|notimestamps|><|emotion_recognition|>'
            query = f"<audio>{audio_path}</audio>{prompt}"
            return audio_path, query, target
        else:
            transcripts_prompt = f"The transcription is: {transcripts}\n"
            if self.gender:
                gender = self.data['gender'][idx]
                if gender == 0:
                    gender_str = "female"
                else:
                    gender_str = "male"
                gender_prompt = f"The gender of the speaker is: {gender_str}\n"
                if self.phones:
                    phones = self.data['phones'][idx]
                    phones_prompt = f"Audio phonemes: {phones}\n"
                    query = f"""Instruction: \nGiven the following audio information and the transcription, predict the emotion of the speaker.\nThe emotion can be one of the following: [ neutral, sad, angry, happy, surprise, fear, disgust, contempt]\n{transcripts_prompt}{gender_prompt}{phones_prompt}Audio information:\n\nOutput:\nemotion ="""
                else:
                    query=f"""Instruction: \nGiven the following audio information and the transcription, predict the emotion of the speaker.\nThe emotion can be one of the following: [ neutral, sad, angry, happy, surprise, fear, disgust, contempt]\n{transcripts_prompt}{gender_prompt}Audio information:\n\nOutput:\nemotion ="""
            else:
                if self.phones:
                    phones = self.data['phones'][idx]
                    phones_prompt = f"Audio phonemes: {phones}\n"
                    query = f"""Instruction: \nGiven the following audio information and the transcription, predict the emotion of the speaker.\nThe emotion can be one of the following: [ neutral, sad, angry, happy, surprise, fear, disgust, contempt]\n{transcripts_prompt}{phones_prompt}Audio information:\n\nOutput:\nemotion ="""
                else:
                    query=f"""Instruction: \nGiven the following audio information and the transcription, predict the emotion of the speaker.\nThe emotion can be one of the following: [ neutral, sad, angry, happy, surprise, fear, disgust, contempt]\n{transcripts_prompt}Audio information:\n\nOutput:\nemotion ="""
            text_tokens = self.tokenizer(query)['input_ids']

            if self.audio_path:
                return audio_path, text_tokens, target
            else:
                y, sr = librosa.load(audio_path, sr=16000)
                if self.audio_encoder_name=='emotion2vec':
                    input_features = load_audio_text_image_video(audio_path, fs=16000, audio_fs=16000,
                                                data_type="sound", tokenizer=self.audio_tokenizer)
                else:
                    inputs = self.feature_extractor(y, sampling_rate=sr, return_tensors="np")
                    input_features = torch.tensor(inputs['input_features'])
                if not self.predict:
                    return input_features, text_tokens, target
                else:
                    return input_features, text_tokens

def collate_fn(batch):
    input_features, text_tokens, target = zip(*batch)
    try:
        audio_inputs=torch.stack(input_features).squeeze(1)
    except TypeError:
        audio_inputs = input_features
    text_tokens = [torch.tensor(text_token) for text_token in text_tokens]
    return [audio_inputs, text_tokens, torch.tensor(target)]

def collate_fn_emo2vec(batch):
    x, text_tokens, target = zip(*batch)
    x = list(x)
    max_len = max([len(i) for i in x])
    mask = torch.zeros(len(x), max_len)
    for i in range(len(x)):
        mask[i, :len(x[i])] = 1
        x[i] = torch.nn.functional.pad(x[i], (0, max_len - len(x[i])))
    audio_inputs = torch.stack(x)
    text_tokens = [torch.tensor(text_token) for text_token in text_tokens]
    return [audio_inputs, mask, text_tokens, torch.tensor(target)]

def collate_fn_predict(batch):
    input_features, text_tokens = zip(*batch)
    try:
        audio_inputs=torch.stack(input_features).squeeze(1)
    except TypeError:
        audio_inputs = input_features
    text_tokens = [torch.tensor(text_token) for text_token in text_tokens]
    return [audio_inputs, text_tokens]

def collate_fn_qwen(batch):
    audio_paths, queries, targets = zip(*batch)
    return {'audio_paths':audio_paths, 
            'queries':queries, 
            'targets':torch.tensor(targets)
            }

class EmoSPeechDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 1)
        self.shuffle = kwargs.get('shuffle', False)
        self.fold = kwargs.get('fold', 0)
        self.train_val = kwargs.get('train_val', False)
        self.predict = kwargs.get('predict', False)
        self.audio_encoder_name = kwargs.get('audio_encoder_name', 'whisper-large-v3')
        if self.audio_encoder_name=='emotion2vec':
            self.collator = collate_fn_emo2vec
        else:
            self.collator = collate_fn

    def setup(self, stage=None):
        train_df = pd.read_csv("data/EmoSPeech_phase_2_train.csv")
        train_df["path"] = train_df["id"].apply(lambda x: f"train_segments/{x}")
        gender = pd.read_csv("data/EmoSPeech_phase_2_train_gender.csv")
        train_df['gender'] = gender["gender"]
        train_df['phones'] = pd.read_csv("data/EmoSPeech_phase_2_train_phonemes.csv")["phonemes"]
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        df_folds = {}
        for i, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
            df_folds[i] = {'train': train_index, 'val': test_index}

        train = train_df.iloc[df_folds[self.fold]['train']].reset_index(drop=True)
        val = train_df.iloc[df_folds[self.fold]['val']].reset_index(drop=True)
        self.train_data = EmoSPeechDataset(train, **self.config)
        self.val_data = EmoSPeechDataset(val, **self.config)
        if not self.predict:
            self.test_data = EmoSPeechDataset(val, **self.config)
        else:
            test_df = pd.read_csv("data/EmoSPeech_phase_2_test_public.csv")
            test_df["path"] = test_df["id"].apply(lambda x: f"test_segments/{x}")
            self.test_data = EmoSPeechDataset(test_df, **self.config)

    def train_dataloader(self):
        if not self.train_val:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collator
            )
        else:
            train_val = ConcatDataset([self.train_data, self.val_data])
            return DataLoader(
                train_val,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collator
                )


    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator
        )
    
    def test_dataloader(self):
        if self.train_val:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_predict
            )
        else:
            return DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_predict
            )

if __name__=='__main__':
    #db_o = OdysseyDataModule()
    #db_o.setup()
    #train = db_o.train_dataloader()
    #for batch in train:
    #    print(batch)
    #    break

    db = EmoSPeechDataModule(fold=0)
    db.setup()
    train = db.train_dataloader()
    for batch in train:
        print(batch)
        break
