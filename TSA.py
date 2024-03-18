import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.metrics.pairwise import manhattan_distances
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
tweet_data = pd.read_csv("/kaggle/input/tweet-data/train.xlsx - Sheet1.csv") 

# Data Preparation
def create_tweet_pairs(data):
  same_user_pairs = []
  diff_user_pairs = []
  for _, group in data.groupby('user'):
    pairs = list(zip(group['text'].values[:-1], group['text'].values[1:]))
    same_user_pairs.extend(pairs)
    if len(pairs) > 1:
      diff_user_pairs.extend([(pairs[i][0], pairs[i+1][1]) for i in range(len(pairs)-1)])
  return same_user_pairs, diff_user_pairs

same_user_pairs, diff_user_pairs = create_tweet_pairs(tweet_data)
pairs = [(pair, 1) for pair in same_user_pairs] + [(pair, 0) for pair in diff_user_pairs]
pairs_df = pd.DataFrame(pairs, columns=['pair', 'label'])

# Data Preprocessing
import spacy

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
  # Tokenize 
  doc = nlp(text)
  # Lemmatize, remove stopwords and punctuation
  tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
  return ' '.join(tokens)

pairs_df['pair'] = pairs_df['pair'].apply(lambda x: tuple(map(preprocess_text, x)))


# Model Architecture
class TweetDataset(Dataset):
  def __init__(self, pairs, tokenizer, max_length):
    self.pairs = pairs
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    text_pair, label = self.pairs[idx]
    encoding = self.tokenizer(text_pair[0], text_pair[1], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, label

class TweetSimilarityModel(nn.Module):
  def __init__(self, transformer_model, hidden_size):
    super(TweetSimilarityModel, self).__init__()
    self.encoder1 = transformer_model
    self.encoder2 = transformer_model
    self.dense = nn.Linear(hidden_size * 2, 1) 
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_ids, attention_mask):
    outputs1 = self.encoder1(input_ids=input_ids[:, 0, :], attention_mask=attention_mask)  
    outputs2 = self.encoder2(input_ids=input_ids[:, 1, :], attention_mask=attention_mask)  
    last_hidden_state1 = outputs1.last_hidden_state
    last_hidden_state2 = outputs2.last_hidden_state

    # Concatenate CLS token representations
    tweet_representations = torch.cat((last_hidden_state1[:, 0, :], last_hidden_state2[:, 0, :]), dim=1)
    similarity_score = self.sigmoid(self.dense(tweet_representations))
    return similarity_score

#Evaluation
def test_model(model, test_data):
 tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 test_dataset = TweetDataset(test_data['pair'], tokenizer, MAX_LENGTH)
 test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  

 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model.to(device)

 with torch.no_grad():
   model.eval()
   y_pred = []
   for batch_idx, data in enumerate(test_dataloader):
     inputs, labels = data
     inputs = inputs.to(device)

     outputs = model(inputs['input_ids'], inputs['attention_mask'])
     y_pred.extend(outputs.squeeze().tolist())

# Clip predictions to [0, 1]
 y_pred = np.clip(y_pred, 0, 1)  
 precision = precision_score(test_data['label'], np.round(y_pred))
 recall = recall_score(test_data['label'], np.round(y_pred))
 f1 = f1_score(test_data['label'], np.round(y_pred))
print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")
