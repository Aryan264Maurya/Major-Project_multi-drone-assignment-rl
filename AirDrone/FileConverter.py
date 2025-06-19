from PIL import Image
import os

def batch_convert_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        # Check for .ppm or .pgm files (case-insensitive)
        if filename.lower().endswith('.ppm') or filename.lower().endswith('.pgm'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Create the new filename with .png extension
            new_filename = os.path.splitext(filename)[0] + '.png'
            save_path = os.path.join(output_dir, new_filename)

            # Save the image as .png
            img.save(save_path)
            print(f"Converted: {filename} -> {new_filename}")

# Example usage
input_dir = 'E:/THU_Projects/DataSets/2025-03-22-20-29-50/images'
output_dir = 'E:/THU_Projects/DataSets/2025-03-22-20-29-50/PNG'
batch_convert_images(input_dir, output_dir)