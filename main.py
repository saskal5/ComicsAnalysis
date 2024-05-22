
import os
import csv
import easyocr
import numpy as np
from PIL import Image

from model import load_trocr_processor, load_trocr_model, load_char_model, run_char_model, load_balloon_model,load_expression_model
from functions import sort_key, delete_files_in_folder
from operations import operations

trocr_processor = load_trocr_processor()
trocr_model = load_trocr_model()

# This needs to run only once to load the model into memory
reader = easyocr.Reader(['en'])

# TO OVERLAY
masked_output_folder = "/output_folder"
#original_image_path_to_overlay = "/output_folder/" + desired_img
cropped_images_folder = "/output_folder/images_cropped/cropped"
output_path_to_overlay = "output_folder/images_cropped/result/result_overlay.png"
output_image_path = "/output_folder/images_cropped/result/char_result_overlay.png"

# CROP ORIGINAL IMAGE
#original_image_path = "/output_folder/images_cropped/original/" + desired_img
output_folder = "/output_folder/images_cropped/original/"

book = 'T_15'
panels_folder = "/Users/azatsaskal/Documents/UGent/Masters Thesis/PDFs_HQ/"

def main():
    folder_url = panels_folder + book
    files = os.listdir(folder_url)
    files.remove('.DS_Store')
    files.remove('Pages')
        
    files = sorted(files, key=sort_key)

    # File path to save the CSV
    csv_file = "t_15.csv"

    # Writing data to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['book', 'page', 'img', 'panel_area', 'panel_width', 'panel_length', 'balloon_to_panel_ratio',
                                                  'detected_chars', 'character_model_confidence', 'balloon_model_confidence', 'detected_balloon_numbers', 
                                                  'associated_balloon_numbers', 'associated_characters', 'text'])
    
        page_nr = 1
        # Write header
        writer.writeheader()
        for file in files:
            original_image_path_to_overlay = "/output_folder/" + file
            
            url = panels_folder + book + "/" + file
            original_image_path = url
            img = Image.open(url)

            if ((img.size[0] < 100) & (img.size[1] < 100)) & ((img.size[0] >= 45) & (img.size[1] >= 45)) &(abs(img.size[0] - img.size[1]) <= 5):
                black_pixels_ratio = np.sum(np.array(img) == 0) / ((np.sum(np.array(img) == 0) + np.sum(np.array(img) == 255)))
                if black_pixels_ratio > 0.25:
                    continue;
                else:
                    page_nr += 1
                    continue;
            elif img.size[0] * img.size[1] < 40000:
                continue;
        
            ## CHARACTER MODEL
            char_model = load_char_model()
            character_result = run_char_model(char_model, book, file)

            ## SPEECH BALLOON MODEL
            balloon_model = load_balloon_model()

            ## EXPRESSION MODEL
            expression_model = load_expression_model()

            ## OPERATIONS
            det_chars, ch_conf, b_conf, det_balloon_nr, ass_balloon_nrs, ass_chars, all_texts, b_to_panel_rat, p_width, p_length, p_area = operations(original_image_path, original_image_path_to_overlay, file, book, char_model, balloon_model, character_result, cropped_images_folder, output_path_to_overlay, output_image_path, output_folder, masked_output_folder, reader, expression_model, trocr_processor, trocr_model)
            
            delete_files_in_folder("/output_folder/images_cropped/original")
            delete_files_in_folder("/output_folder/images_cropped/textboxes")
            delete_files_in_folder("/output_folder/images_cropped/result")
            delete_files_in_folder("/output_folder/images_cropped/cropped/")

            row = {"book": book, 
                "page": page_nr,
                "img": file, 
                "panel_area": p_area,
                "panel_width": p_width,
                "panel_length": p_length,
                "balloon_to_panel_ratio": b_to_panel_rat,
                "detected_chars": det_chars,
                "character_model_confidence": ch_conf,
                "balloon_model_confidence": b_conf,
                "detected_balloon_numbers": det_balloon_nr,
                "associated_balloon_numbers": ass_balloon_nrs,
                "associated_characters": ass_chars, 
                "text": all_texts}
            
            writer.writerow(row)
        print(f"Data has been written to {csv_file}")

if __name__ == "__main__":
    main()      
