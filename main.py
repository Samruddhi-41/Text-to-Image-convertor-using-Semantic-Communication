import argparse  # Module for command-line argument parsing
from image_generator import ImageGenerator  # Import our custom image generator class
from config import Config  # Import configuration settings
import os  # Module for interacting with the operating system (file paths, etc.)

def main():
    # Create an argument parser to handle command-line inputs
    parser = argparse.ArgumentParser(description="Semantic Text-to-Image Generation")
    
    # Define the arguments that the program can accept
    parser.add_argument("--prompt", type=str, required=True, 
                       help="Text description for image generation")  # Text prompt is required
    
    parser.add_argument("--num_images", type=int, default=1, 
                       help="Number of images to generate")  # Default: generate 1 image
    
    parser.add_argument("--variation", type=str, 
                       help="Path to image for generating variations")  # Optional: path to reference image
    
    parser.add_argument("--num_variations", type=int, default=1, 
                       help="Number of variations to generate")  # Default: generate 1 variation
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Initialize the image generator object
    generator = ImageGenerator()
    
    try:
        # Check if we're in variation mode (modifying an existing image)
        if args.variation:
            # Verify that the specified image file exists
            if not os.path.exists(args.variation):
                print(f"Error: Image file not found at {args.variation}")
                return
            
            # Inform the user about the variation generation process
            print(f"Generating {args.num_variations} variations of {args.variation}...")
            
            # Call the method to generate variations of the provided image
            saved_paths = generator.generate_variations(
                args.variation,  # Path to the input image
                args.prompt,     # Text description for guiding the variation
                args.num_variations  # Number of variations to create
            )
        else:
            # We're in regular text-to-image mode
            # Inform the user about the image generation process
            print(f"Generating {args.num_images} images from prompt: {args.prompt}")
            
            # Call the method to generate images from the text prompt
            saved_paths = generator.generate_image(
                args.prompt,     # Text description for image generation
                args.num_images  # Number of images to create
            )
        
        # Display the paths where the generated images were saved
        print("\nGenerated images saved at:")
        for path in saved_paths:
            print(f"- {path}")
            
    except Exception as e:
        # Catch and display any errors that occur during the generation process
        print(f"Error during image generation: {str(e)}")

# Standard Python idiom to check if this script is being run directly (not imported)
if __name__ == "__main__":
    main()  # Call the main function if script is run directly 