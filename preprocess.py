from src.utils import download_image
import pandas as pd
from pytesseract import pytesseract
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def process_single_image(row, folder_path):
    index, image_link = row.name, row['image_link']
    image_filename = f'{image_link[-15:-4]}.jpg'
    image_path = os.path.join(folder_path, image_filename)

    print(f'Processing index {index}: Downloading image from {image_link}')
    download_image(image_link, folder_path)

    text = ''
    if os.path.exists(image_path):
        try:
            print(f'Opening image: {image_path}')
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            print(f"Extracted text for index {index}: {text[:100]}...")
        except Exception as e:
            print(f'Error processing image {image_path}: {e}')
    else:
        print(f'Image not found: {image_path}')

    return index, text


def process_images_multithreaded(df, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    text_dict = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(lambda row: process_single_image(row, folder_path), [row for _, row in df.iterrows()])

        for index, text in results:
            text_dict[index] = text

    return text_dict


df = pd.read_csv('train_split.csv')
df=df[:4000]
text_data = process_images_multithreaded(df, 'train_images')

# Create a new DataFrame with the extracted text
df_text = pd.DataFrame({'image_link': df['image_link'], 'Text': [text_data.get(i, '') for i in df.index]})

# Save the new DataFrame to train_text.csv in the dataset folder
output_folder = 'dataset'
os.makedirs(output_folder, exist_ok=True)
df_text.to_csv(os.path.join(output_folder, 'train_text.csv'), index=False, encoding='utf-8')

print("CSV with text saved successfully.")
