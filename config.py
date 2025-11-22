import torch
import os

class Config:
    # Model settings
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # Smaller model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization settings
    ENABLE_ATTENTION_SLICING = True
    ENABLE_VAE_SLICING = True
    ENABLE_VAE_TILING = True
    ENABLE_SEQUENTIAL_CPU_OFFLOAD = False
    ENABLE_MODEL_CPU_OFFLOAD = False
    
    # Generation parameters
    NUM_INFERENCE_STEPS = 30  # Reduced steps
    GUIDANCE_SCALE = 7.5
    IMAGE_SIZE = 512
    
    # Semantic processing
    MAX_PROMPT_LENGTH = 77
    SEMANTIC_ENHANCEMENT = True
    
    # Output settings
    OUTPUT_DIR = "generated_images"
    SAVE_FORMAT = "png"
    
    # Cache settings
    CACHE_DIR = "model_cache"
    
    @staticmethod
    def get_generation_config():
        return {
            "num_inference_steps": Config.NUM_INFERENCE_STEPS,
            "guidance_scale": Config.GUIDANCE_SCALE,
            "width": Config.IMAGE_SIZE,
            "height": Config.IMAGE_SIZE,
        } 