from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import torch
import numpy as np
import pickle

string_to_remove = "الحمد لله والصلاة والسلام على رسول الله وعلى آله وصحبه أما بعد:"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSEmbedding:
    def __init__(self, model_name, token):
        logging.info("Initializing FAISS embedding class")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        
        # Check if the tokenizer has a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True)
        
        # Resize model embeddings if new tokens were added
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.index = None
        self.embeddings = None
        self.titles = []
        self.questions = []
        self.answers = []
        self.data= None

    def load_csv_data(self, file_path):
        logging.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path, encoding='utf-8')
        if 'ques' not in self.data.columns or 'ans' not in self.data.columns:
            logging.error("CSV file must contain 'ques' and 'ans' columns")
            raise ValueError("CSV file must contain 'ques' and 'ans' columns")
        
        # Convert non-string columns to strings
        self.data['ques'] = self.data['ques'].astype(str)
        self.data['ans'] = self.data['ans'].astype(str)
        if 'title' in self.data.columns:
            self.data['title'] = self.data['title'].astype(str)
        
        # select first 1k rows
        self.data = self.data[:200]
        self.data = FAISSEmbedding.clean_data(self.data)
       
        # clean data 
        self.data['ans'] = self.data['ans'].apply(FAISSEmbedding.remove_text)
        
        self.questions = self.data['ques'].tolist()
        self.answers = self.data['ans'].tolist()
        if 'title' in self.data.columns:
            self.titles = self.data['title'].tolist()
        logging.info(f"Data loaded successfully, shape: {self.data.shape}")
        print(self.data.head())
    
    @staticmethod
    def clean_data(data: pd.DataFrame):
        # Remove any rows with missing values
        data = data.dropna()
        # Remove any duplicate rows
        data = data.drop_duplicates()
        return data
    
    @staticmethod
    def remove_text(text:str):
        if text.startswith(string_to_remove):
            return text[len(string_to_remove):].strip()
        return text

    
    
    def get_embeddings_in_batches(self,texts, batch_size=32, type:str = 'titles'):
        logging.info(f"Creating embeddings for the {len(texts)} {type} in batches of {batch_size}")
        
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden state
            batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
            embeddings_list.append(batch_embeddings)
        print(embeddings_list)
        return np.vstack(embeddings_list)
    
    def create_embeddings(self, batch_size=8):
        title_embeddings = self.get_embeddings_in_batches(self.titles, batch_size, type= 'titles')
        ques_embeddings = self.get_embeddings_in_batches(self.questions, batch_size,type= 'questions')
        ans_embeddings = self.get_embeddings_in_batches(self.answers, batch_size,type= 'answers')
        
        # Combine embeddings into a single array
        self.embeddings = np.hstack((title_embeddings, ques_embeddings, ans_embeddings))
        logging.info(f"Embeddings created, shape: {self.embeddings.shape}")

    def build_index(self):
        logging.info("Building FAISS index")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logging.info("FAISS index built")

    def save_index(self, file_path, metadata_path):
        logging.info(f"Saving FAISS index to {file_path}")
        faiss.write_index(self.index, file_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({'titles': self.titles, 'questions': self.questions, 'answers': self.answers}, f)
        logging.info("FAISS index and metadata saved")

    def load_index(self, file_path, metadata_path):
        logging.info(f"Loading FAISS index from {file_path}")
        self.index = faiss.read_index(file_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.titles = metadata['titles']
            self.questions = metadata['questions']
            self.answers = metadata['answers']
        logging.info("FAISS index and metadata loaded")

    def search(self, query, top_k=5):
        logging.info(f"Searching for top {top_k} similar questions for the query: '{query}'")
        
        # Generate the embedding for the query
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden state
        query_embedding = hidden_states.mean(dim=1).cpu().numpy()
        
        # Ensure the query vector matches the combined embedding dimensions
        query_vector = np.hstack((query_embedding, query_embedding, query_embedding)).reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        most_similar_entries = []
        num_entries = len(self.questions)  # Number of entries in the dataset
        
        for i in indices[0]:
            if i < num_entries:  # Check if index is within valid range
                entry = {
                    "title": self.titles[i] if self.titles else "",
                    "question": self.questions[i],
                    "answer": self.answers[i],
                    # "distance": distances[i]
                }
                most_similar_entries.append(entry)
            else:
                logging.warning(f"Index {i} is out of bounds for the dataset with size {num_entries}")
        
        logging.info(f"Search completed for query: '{query}'")
        logging.info(f"Results: {most_similar_entries}")
        logging.info(f"Distances: {distances}")
        
        return most_similar_entries
