from transformers import AutoTokenizer, AutoModel  # Import Hugging Face transformers components
import torch  # PyTorch library for deep learning operations
import numpy as np  # NumPy for numerical operations
from config import Config  # Import our configuration settings

class SemanticProcessor:
    """
    The SemanticProcessor class is the core of the semantic communication system.
    It uses BERT to understand the semantic meaning of text and enhance prompts
    for better image generation. This represents the "semantic encoder" in the
    semantic communication framework.
    """
    
    def __init__(self):
        """
        Initialize the semantic processor with BERT model and tokenizer.
        BERT (Bidirectional Encoder Representations from Transformers) provides
        contextualized word embeddings that capture semantic meaning.
        """
        # Load the BERT tokenizer - converts text to token IDs that BERT understands
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load the BERT model - provides contextual understanding of language
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Move the model to GPU if available for faster processing
        self.model.to(Config.DEVICE)
        
        # Set model to evaluation mode (disables dropout for deterministic results)
        self.model.eval()
    
    def get_semantic_embedding(self, text):
        """
        Get semantic embedding for the input text.
        
        This function extracts the semantic meaning of text as a vector representation.
        These embeddings capture the deep semantic content beyond just the words.
        
        Args:
            text: Input text to encode semantically
            
        Returns:
            Numpy array containing the semantic embedding vector
        """
        # Tokenize the text and prepare it for the model
        # - return_tensors="pt": Return PyTorch tensors
        # - padding=True: Add padding for consistent lengths
        # - truncation=True: Truncate texts longer than max_length
        # - max_length: Maximum token length from config
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=Config.MAX_PROMPT_LENGTH)
        
        # Move inputs to the same device as the model (GPU/CPU)
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
        
        # Process the text through BERT without computing gradients (inference only)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the [CLS] token embedding which represents the entire sentence
        # - last_hidden_state contains embeddings for each token
        # - [:, 0, :] selects the first token ([CLS]) for all items in the batch
        # - .cpu().numpy() moves the result to CPU and converts to numpy array
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    def enhance_prompt(self, prompt):
        """
        Enhance the prompt using semantic understanding.
        
        This function represents the core of semantic enhancement in the system.
        It takes a user prompt and enhances it with additional semantic context
        to produce better image generation results.
        
        Args:
            prompt: Original text prompt from the user
            
        Returns:
            Enhanced prompt with additional semantic context
        """
        # Skip enhancement if disabled in config
        if not Config.SEMANTIC_ENHANCEMENT:
            return prompt
            
        # Get semantic embedding for understanding the prompt's meaning
        embedding = self.get_semantic_embedding(prompt)
        
        # In a more advanced implementation, we would analyze the embedding
        # to determine appropriate enhancements based on semantic content.
        # For now, we add general quality boosting terms.
        enhanced_prompt = f"{prompt}, highly detailed, photorealistic, 4k"
        
        return enhanced_prompt
    
    def get_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.
        
        This function compares the semantic meaning of two texts to determine
        how similar they are in meaning (not just words). This is useful for
        evaluating if generated images match the intended meaning.
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            
        Returns:
            Similarity score (0-1) where 1 means identical meaning
        """
        # Get semantic embeddings for both texts
        emb1 = self.get_semantic_embedding(text1)
        emb2 = self.get_semantic_embedding(text2)
        
        # Calculate cosine similarity between the embeddings
        # Cosine similarity measures the angle between vectors, indicating
        # how similar their directions are in the semantic space
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Return the similarity score (scalar value)
        return similarity[0][0] 