# data/transforms.py

import torch
import random
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import random
from nltk.corpus import wordnet

class TextTransforms:
    @staticmethod
    def random_deletion(text, p=0.05):
        """Randomly delete words from text with probability p."""
        words = text.split()
        if len(words) <= 3:  # Don't delete if too few words
            return text
            
        # Delete words with probability p
        result = [word for word in words if random.random() > p]
        
        # Make sure we don't delete all words
        if not result:
            return text
            
        return ' '.join(result)
    
    @staticmethod
    def synonym_replacement(text, p=0.2):
        """Replace words with synonyms with probability p using WordNet."""
        words = text.split()
        if len(words) <= 3:  # Don't replace if too few words
            return text
            
        for i in range(len(words)):
            if random.random() < p:
                synonyms = []
                for syn in wordnet.synsets(words[i]):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != words[i]:
                            synonyms.append(synonym)
                            
                if len(synonyms) > 0:
                    words[i] = random.choice(synonyms)
                    
        return ' '.join(words)
    
    @staticmethod
    def identity(text):
        """No change to text."""
        return text


class RandomTextTransform:
    """Class-based transform that's pickle-friendly."""
    def __init__(self, transforms, p=0.5):
        """
        Args:
            transforms: List of transform functions to apply
            p: Probability of applying a transform (vs identity)
        """
        self.transforms = transforms
        self.p = p
        
    def __call__(self, text):
        if random.random() > self.p:
            return text  # No transformation
        
        # Choose a random transform
        transform_func = random.choice(self.transforms)
        return transform_func(text)


# Backtranslation for dataset doubling
class BackTranslator:
    def __init__(self, src_lang="en", tgt_lang="fr", device=None):
        """
        Initialize back translation model.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code for intermediate translation
            device: Device to run models on (default: cuda if available, else cpu)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading translation models on {self.device}...")
        # Source to target model
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.src_to_tgt_model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self.src_to_tgt_tokenizer = MarianTokenizer.from_pretrained(model_name)
        
        # Target to source model
        model_name = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
        self.tgt_to_src_model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self.tgt_to_src_tokenizer = MarianTokenizer.from_pretrained(model_name)
        print("Translation models loaded successfully.")
    

    def translate(self, texts, batch_size=8):
        """
        Translate a list of texts from source to target language.
        
        Args:
            texts: List of texts to translate
            batch_size: Batch size for translation
        
        Returns:
            List of translated texts
        """
        results = []
        
        # Create a progress bar for all batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc=f"Translating {self.src_lang}→{self.tgt_lang}", 
                unit="batch", dynamic_ncols=True) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Tokenize texts
                inputs = self.src_to_tgt_tokenizer(batch, return_tensors="pt", padding=True, 
                                                truncation=True, max_length=512).to(self.device)
                
                # Translate
                with torch.no_grad():
                    translated = self.src_to_tgt_model.generate(**inputs)
                
                # Decode translations
                translations = self.src_to_tgt_tokenizer.batch_decode(translated, skip_special_tokens=True)
                results.extend(translations)
                
                # Update progress bar
                pbar.update(1)
                
        return results

    def back_translate(self, texts, batch_size=8):
        """
        Back-translate a list of texts (source -> target -> source).
        
        Args:
            texts: List of texts to back-translate
            batch_size: Batch size for translation
            
        Returns:
            List of back-translated texts
        """
        # First translation (source -> target)
        print(f"Step 1: Translating {len(texts)} texts from {self.src_lang} to {self.tgt_lang}")
        intermediate = self.translate(texts, batch_size)
        
        # Second translation (target -> source)
        print(f"Step 2: Translating back from {self.tgt_lang} to {self.src_lang}")
        results = []
        
        # Create a progress bar for the back translation
        total_batches = (len(intermediate) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc=f"Translating {self.tgt_lang}→{self.src_lang}", 
                unit="batch", dynamic_ncols=True) as pbar:
            for i in range(0, len(intermediate), batch_size):
                batch = intermediate[i:i+batch_size]
                
                # Tokenize intermediate texts
                inputs = self.tgt_to_src_tokenizer(batch, return_tensors="pt", padding=True, 
                                                truncation=True, max_length=512).to(self.device)
                
                # Translate back
                with torch.no_grad():
                    back_translated = self.tgt_to_src_model.generate(**inputs)
                
                # Decode translations
                back_translations = self.tgt_to_src_tokenizer.batch_decode(back_translated, skip_special_tokens=True)
                results.extend(back_translations)
                
                # Update progress bar
                pbar.update(1)
        
        return results

