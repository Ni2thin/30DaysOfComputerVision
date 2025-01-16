import os
from PIL import Image
import random


def make_index(logos):
    """
    Creates dictionaries for mapping between logo names and indices.
    """
    index2logo = {i: logo for i, logo in enumerate(logos)}
    logo2index = {logo: i for i, logo in enumerate(logos)}
    return index2logo, logo2index


def create_syntitized_img(image, logo, output_path):
    """
    Places a logo onto a background image at a random size and position.
    Ensures the logo size is smaller than the background and handles random positioning.
    """
    # Open the background and logo images
    background = Image.open(image).convert("RGB")  # Ensure background is RGB
    foreground = Image.open(logo).convert("RGBA")  # Ensure logo is RGBA

    # Get background dimensions
    max_width, max_height = background.size

    # Resize the logo to fit within the background dimensions
    scale = random.uniform(0.25, 0.5)  # Random scaling factor (adjustable)
    foreground_width, foreground_height = foreground.size
    resized_width = int(foreground_width * scale)
    resized_height = int(foreground_height * scale)

    # Ensure the resized logo fits within the background dimensions
    if resized_width > max_width or resized_height > max_height:
        scale = min(max_width / foreground_width, max_height / foreground_height) * 0.8  # Adjust scale
        resized_width = int(foreground_width * scale)
        resized_height = int(foreground_height * scale)

    # Resize the logo
    foreground = foreground.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    foreground_width, foreground_height = resized_width, resized_height

    # Generate random position for the logo
    x = random.randint(0, max_width - foreground_width)
    y = random.randint(0, max_height - foreground_height)

    # Create a mask for the transparent regions of the logo
    mask = foreground.split()[3]  # Extract the alpha channel as a mask

    # Paste the logo onto the background using the transparency mask
    background.paste(foreground, (x, y), mask)

    # Save the resulting image
    background.save(output_path)

    # Compute YOLO annotation parameters
    center_x = (x + foreground_width / 2) / max_width
    center_y = (y + foreground_height / 2) / max_height
    norm_width = foreground_width / max_width
    norm_height = foreground_height / max_height

    return center_x, center_y, norm_width, norm_height



def save_syntitized_images(output_dir='train'):
    """
    Generates syntitized images and YOLO annotations for the specified output directory.
    """
    base_dir = '/Users/nitthin/Downloads/Invoice-Logo-Detection-main'
    images_path = os.path.join(base_dir, 'images')
    logos_path = os.path.join(base_dir, 'logos')

    images = os.listdir(images_path)  # List of images
    logos = os.listdir(logos_path)  # List of logos

    _, logo2index = make_index(logos)

    images_output_dir = os.path.join(base_dir, f'images1/{output_dir}')
    labels_output_dir = os.path.join(base_dir, f'labels/{output_dir}')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    i = 0  # Image pointer
    j = 0  # Logo pointer

    while i < len(images):
        if j == len(logos):  # Reset logo pointer if all logos are used
            j = 0

        img_path = os.path.join(images_path, images[i])
        logo_path = os.path.join(logos_path, logos[j])
        output_image_path = os.path.join(images_output_dir, images[i])
        output_label_path = os.path.join(labels_output_dir, f"{os.path.splitext(images[i])[0]}.txt")

        if not os.path.exists(output_image_path):
            # Generate syntitized image and YOLO annotation
            data = create_syntitized_img(img_path, logo_path, output_image_path)

            # Save annotation to text file
            label = logo2index[logos[j]]
            center_x, center_y, norm_width, norm_height = data
            with open(output_label_path, 'w') as f:
                f.write(f"{label} {center_x} {center_y} {norm_width} {norm_height}\n")

        i += 1
        j += 1


# Generate training, validation, and test datasets
save_syntitized_images(output_dir='train')
save_syntitized_images(output_dir='val')
save_syntitized_images(output_dir='test')
