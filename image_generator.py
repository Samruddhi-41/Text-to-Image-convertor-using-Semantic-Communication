from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
from config import Config
from semantic_processor import SemanticProcessor

class ImageGenerator:
    def __init__(self):
        """Initialize the image generator with text-to-image pipeline only."""
        try:
            # Initialize with CPU and float32 first
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                Config.MODEL_NAME,
                torch_dtype=torch.float32,
                cache_dir=Config.CACHE_DIR
            )
            
            # Initialize variation properties
            self.variation_strength = 0.75
            
            # Apply memory optimizations
            if Config.ENABLE_ATTENTION_SLICING:
                self.pipeline.enable_attention_slicing()
            if Config.ENABLE_VAE_SLICING:
                self.pipeline.enable_vae_slicing()
            if Config.ENABLE_VAE_TILING:
                self.pipeline.enable_vae_tiling()
            
            # Try GPU if available
            if Config.DEVICE == "cuda":
                try:
                    # Create a new pipeline instance with half precision for GPU
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        Config.MODEL_NAME,
                        torch_dtype=torch.float16,
                        cache_dir=Config.CACHE_DIR
                    ).to("cuda")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print("GPU error, falling back to CPU")
                    Config.DEVICE = "cpu"
                    # Keep the original float32 pipeline on CPU
                    self.pipeline = self.pipeline.to("cpu")
            else:
                self.pipeline = self.pipeline.to(Config.DEVICE)
            
            self.semantic_processor = SemanticProcessor()
            
            # Create output directory if it doesn't exist
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            
        except Exception as e:
            print(f"Error initializing pipeline: {str(e)}")
            raise
    
    def generate_image(self, prompt, num_images=1):
        """Generate image from text prompt"""
        try:
            # Enhance prompt using semantic understanding
            enhanced_prompt = self.semantic_processor.enhance_prompt(prompt)
            
            print(f"Generating image with prompt: {enhanced_prompt}")
            print(f"Device: {Config.DEVICE}, Steps: {Config.NUM_INFERENCE_STEPS}")
            
            # Generate images
            images = self.pipeline(
                enhanced_prompt,
                **Config.get_generation_config(),
                num_images_per_prompt=num_images
            ).images
            
            # Save images
            saved_paths = []
            for i, image in enumerate(images):
                # Create safe filename
                safe_prompt = "".join(c if c.isalnum() or c in [' ', '_'] else '_' for c in prompt[:30])
                safe_prompt = safe_prompt.replace(' ', '_')
                
                filename = f"{safe_prompt}_{i}.{Config.SAVE_FORMAT}"
                filepath = os.path.join(Config.OUTPUT_DIR, filename)
                image.save(filepath)
                saved_paths.append(filepath)
            
            return saved_paths
            
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory during generation, trying with reduced batch size")
            # Try with single image if batch generation failed
            if num_images > 1:
                return self.generate_image(prompt, num_images=1)
            raise
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e) or "CUDA out of memory" in str(e):
                print("Error during generation, falling back to CPU")
                Config.DEVICE = "cpu"
                # Reinitialize the pipeline for CPU
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    Config.MODEL_NAME,
                    torch_dtype=torch.float32,
                    cache_dir=Config.CACHE_DIR
                ).to("cpu")
                return self.generate_image(prompt, num_images)
            raise
    
    def generate_variations(self, image_path, prompt, num_variations=1):
        """Generate variations of an existing image using img2img technique"""
        try:
            # Load the input image
            init_image = Image.open(image_path).convert("RGB")
            
            # Resize image to match model requirements
            width, height = init_image.size
            if width != height:
                size = min(width, height)
                init_image = init_image.crop(((width - size) // 2,
                                             (height - size) // 2,
                                             (width + size) // 2,
                                             (height + size) // 2))
            init_image = init_image.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            # Enhance prompt using semantic understanding
            enhanced_prompt = self.semantic_processor.enhance_prompt(prompt)
            
            print(f"Generating variation with prompt: {enhanced_prompt}")
            print(f"Device: {Config.DEVICE}, Steps: {Config.NUM_INFERENCE_STEPS}")
            
            # First, generate a negative prompt embedding to act as a starting point
            negative_prompt = "low quality, blurry, distorted"
            
            # Generate a noise image to start with
            generator = torch.Generator(device=Config.DEVICE).manual_seed(42)
            
            # Generate images using the standard pipeline but with a lot more noise
            images = self.pipeline(
                enhanced_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=Config.GUIDANCE_SCALE,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                num_images_per_prompt=num_variations,
                generator=generator
            ).images
            
            # Save variations
            saved_paths = []
            for i, image in enumerate(images):
                # Create a blended image (simple solution without img2img pipeline)
                # This blends the original image with the generated one based on strength
                blended_image = Image.blend(
                    init_image, 
                    image.resize(init_image.size), 
                    self.variation_strength  # Blend factor (0-1)
                )
                
                filename = f"variation_{i}_{os.path.basename(image_path)}"
                filepath = os.path.join(Config.OUTPUT_DIR, filename)
                blended_image.save(filepath)
                saved_paths.append(filepath)
            
            return saved_paths
            
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory during variation generation, trying with single variation")
            # Try with single variation if batch generation failed
            if num_variations > 1:
                return self.generate_variations(image_path, prompt, num_variations=1)
            raise
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e) or "CUDA out of memory" in str(e):
                print("Error during variation generation, falling back to CPU")
                Config.DEVICE = "cpu"
                # Reinitialize the pipeline for CPU
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    Config.MODEL_NAME,
                    torch_dtype=torch.float32,
                    cache_dir=Config.CACHE_DIR
                ).to("cpu")
                return self.generate_variations(image_path, prompt, num_variations)
            raise 