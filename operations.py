
import os
import re
import cv2
import math
import pytesseract

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from shapely.geometry import Polygon

from functions import mask_single_image, draw_boxes, crop_and_save, crop_original_image, save_corners
from functions import overlay_largest_area_contours_with_numbers
from functions import find_contour_corners, save_difference_region, find_tail_point, mark_tail_point
from functions import find_contour_with_corners, save_contour_with_corners
from functions import find_smallest_angle_triangle, mark_triangle, visualize_and_extend
from functions import is_character_intersecting_region, process_groups, find_locations, find_elements_by_indexes

from model import run_expression_model, run_balloon_model

def operations(original_image_path, original_image_path_to_overlay, desired_img, desired_book, char_model, balloon_model,
            character_result, cropped_images_folder, output_path_to_overlay, output_image_path, 
            output_folder, masked_output_folder, reader, expression_model, trocr_processor, trocr_model):
    
    mask_single_image(desired_img, 
                  '/Users/azatsaskal/Documents/UGent/Masters Thesis/PDFs_HQ/' + desired_book + '/', 
                  masked_output_folder, 
                  threshold=220)

    img = Image.open(original_image_path)
    img_gray = Image.open(original_image_path_to_overlay).convert('L')
    img_masked = Image.open('/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/' + desired_img)

    balloon_result = run_balloon_model(balloon_model, desired_img)

    boxes_to_be_kept = []
    for box in balloon_result[0].boxes.xyxy:

        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        coords = (xA, yA, xB, yB)
        img_cropped = img_masked.crop(coords)
        
        #print(np.sum(np.array(img_cropped) == 0))
        #print(np.sum(np.array(img_cropped) == 255))
        
        black_pixels_ratio = np.sum(np.array(img_cropped) == 0) / ((np.sum(np.array(img_cropped) == 0) + np.sum(np.array(img_cropped) >= 240)))
        if black_pixels_ratio > 0.90:
            continue;
        else:
            boxes_to_be_kept.append(box)

    detected_balloons = []

    for r in balloon_result:
            for c in r.boxes.cls:
                detected_balloons.append(int(c))

    character_coords_cropped, balloon_coordinates_list = crop_and_save(character_result, boxes_to_be_kept, img, img_gray)

    threshold_value = 200 #245
    largest_area_contours_list, balloon_to_panel_ratio, panel_length, panel_width, panel_area  = overlay_largest_area_contours_with_numbers(original_image_path, 
                                                                                            cropped_images_folder, 
                                                                                            balloon_coordinates_list, 
                                                                                            threshold_value, 
                                                                                            output_path_to_overlay)
    
    if len(largest_area_contours_list) <= 0:
        balloon_coordinates_list = []

    # Load your image
    overlayed_image = cv2.imread(output_path_to_overlay)

    # Draw bounding boxes on the image
    draw_boxes(overlayed_image, character_coords_cropped)

    # Save the resulting print
    #cv2.imwrite(output_image_path, overlayed_image)

    ## CROP ORIGINAL IMAGE
    coords_crop_from_org_balloons = crop_original_image(original_image_path, largest_area_contours_list, output_folder=output_folder)

    ## FINDING THE TAIL
    smallest_angle_traingles = []
    tail_points = []

    blur_strength = 19
    for balloon_nr in range(len(balloon_coordinates_list)):
        input_cropped_image_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/cropped_image_" + str(balloon_nr+1) + ".png" 
        output_original_corners_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/original_corners_" + str(balloon_nr+1) + ".png"
        output_blurred_corners_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/blurred_corners" + str(balloon_nr+1) + ".png"
        output_difference_region_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/difference_region_" + str(balloon_nr+1) + ".png"
        output_marked_image_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/marked_image" + str(balloon_nr+1) + ".png"
        
        original_cropped_image = cv2.imread(input_cropped_image_path)
        
        # Find and save corners of the blurred image
        original_corners = find_contour_corners(original_cropped_image)
        save_corners(original_cropped_image, original_corners, output_original_corners_path)

        # Blur the original image
        blurred_image = cv2.GaussianBlur(original_cropped_image, (blur_strength, blur_strength), 0)

        # Find and save corners of the blurred image
        blurred_corners = find_contour_corners(blurred_image)
        save_corners(blurred_image, blurred_corners, output_blurred_corners_path)

        # Find and save the region that the original contour covers but the blurred contour does not
        save_difference_region(original_cropped_image, blurred_image, output_difference_region_path)

        # Find the tail point based on the difference region and blurred contour
        tail_point = find_tail_point(original_corners, blurred_corners)
        tail_points.append(tail_point)

        # Mark the tail point on the original image and save
        mark_tail_point(original_cropped_image, tail_point, output_marked_image_path)

        ## TRIANGLE WITH SMALLEST ANGLE
        output_contour_corners_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/contour_corners" + str(balloon_nr + 1) +".png"
        output_marked_triangle_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/marked_triangle" + str(balloon_nr + 1) + ".png"
        
        # Find and save contour with corners of the original image
        contour, corners = find_contour_with_corners(original_cropped_image)
        save_contour_with_corners(original_cropped_image, contour, corners, output_contour_corners_path)

        # Find the set of three corner points with the smallest angle
        smallest_angle_triangle = find_smallest_angle_triangle(corners, tail_point)
        smallest_angle_traingles.append(smallest_angle_triangle)

        # Mark the set of three corner points with the smallest angle on the original image
        mark_triangle(original_cropped_image, smallest_angle_triangle, output_marked_triangle_path)

    # Read the original image
    original_image_to_extend = cv2.imread(original_image_path)
        
    edges_list_all = []
    for balloon_nr in range(len(balloon_coordinates_list)): 
        
        # Specify the paths
        output_visualization_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/extended_lines" + str(balloon_nr+1) + ".png"

        # Coordinates of the contour in the original image corresponding to the cropped image
        contour_coordinates = coords_crop_from_org_balloons[balloon_nr]

        # Visualize the triangle with the corner point, the two supporting edges, and extend both adjacent edges on the original image
        edges_list = visualize_and_extend(original_image_to_extend, contour_coordinates, smallest_angle_traingles[balloon_nr], output_visualization_path)
        
        edges_list_all.append(edges_list)

    # Read the original image
    original_image = cv2.imread(original_image_path)

    def distance(point, bbox):
        # Calculate the center of the bounding box
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Calculate the Euclidean distance between the point and the bbox center
        return math.sqrt((point[0] - bbox_center[0]) ** 2 + (point[1] - bbox_center[1]) ** 2)

    character_bbox_list = character_coords_cropped
    balloon_nr = 0

    character_present_or_not_list = []
    balloon_nr_list = []

    balloon_nrs = []
    associated_chars = []
    associated_chars_dict = {}

    for i, edges_list in enumerate(edges_list_all):
        
        extended_edge1_start = edges_list[0]
        extended_edge2_start = edges_list[1]
        edge_end = edges_list[2]

        # Update edges_list with the correct starting and ending points of the extended lines
        edges_list_updated = [extended_edge1_start, edge_end, edge_end, extended_edge2_start]

        # Visualize the triangle with the corner point, the two supporting edges, and extend both adjacent edges on the original image
        visualize_and_extend(original_image, contour_coordinates, smallest_angle_triangle, output_visualization_path)
        print()
        # Convert the extended lines to a polygon
        extended_lines_polygon = Polygon(edges_list_updated)
        
        # Rotate the polygon by 90 degrees
        #extended_lines_polygon = rotate(extended_lines_polygon, angle=-90, origin='center')

        # Draw the polygon formed by the extended lines on the original image
        cv2.polylines(original_image, [np.array(edges_list)], isClosed=True, color=(0, 255, 255), thickness=2)

        # Assuming character_bbox_list is a list of bounding box coordinates of detected characters
        for character_bbox in character_bbox_list:
            # Draw the bounding box of the character on the original image
            x, y, width, height = character_bbox
            cv2.rectangle(original_image, (x, y), (width, height), (0, 255, 0), 2)
        
        # Assuming character_bbox_list is a list of bounding box coordinates of detected characters   
        character_nr = 0

        ## STORE THE DETECTED CHARACTERS
        detected_chars = []
        names = char_model.names

        for r in character_result:
            for c in r.boxes.cls:
                detected_chars.append(names[int(c)])
        
        intersection_count = 0
        intersected_character_names = []
        intersected_character_boxes = []
        for character_bbox in character_bbox_list:
            if is_character_intersecting_region(character_bbox, extended_lines_polygon):
                intersection_count += 1
                balloon_nrs.append(balloon_nr+1)
                associated_chars.append(detected_chars[character_nr])
                #print("Balloon number", str(balloon_nr+1), " -> ", detected_chars[character_nr])
                character_present_or_not_list.append(detected_chars[character_nr])
                intersected_character_names.append(detected_chars[character_nr])
                intersected_character_boxes.append(character_bbox)
                
            else: 
                character_present_or_not_list.append("NA")
            balloon_nr_list.append(balloon_nr)  
            character_nr += 1

        if intersection_count > 1:
            print("EĞER", i)
            tail_point_to_calc = edges_list[2]
            print("tail_point_to_calc", tail_point_to_calc)

            distances_from_tail_to_box = []
            for it in range(intersection_count):
                distances_from_tail_to_box.append(distance(tail_point_to_calc, intersected_character_boxes[it]))

            print("distances_from_tail_to_box", distances_from_tail_to_box)
            closer_char_ind = distances_from_tail_to_box.index(min(distances_from_tail_to_box))
            closer_char_name = intersected_character_names[closer_char_ind]
            print("i", i, "closer_char_name", closer_char_name)

            associated_chars_dict[i] = closer_char_name
        elif len(intersected_character_names) > 0:
            associated_chars_dict[i] = intersected_character_names[0]
        else:
            associated_chars_dict[i] = "NaN"


        # Save or display the modified image
        cv2.imwrite("/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/balloon_char_association" + str(balloon_nr + 1) + ".png", original_image)
        balloon_nr += 1

    #print(detected_chars)

    # Find if any balloon association is made for more than once
    #duplicate_indices = [i for i, num in enumerate(balloon_nrs) if balloon_nrs.count(num) > 1]
        

    '''
    def store_duplicates(arr):
        count_dict = {}
        duplicates = []

        for num in arr:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1

        for num, count in count_dict.items():
            if count > 1:
                duplicates.append(num)

        return len(duplicates), duplicates

    duplicates_len, duplicates_ind_list = store_duplicates(balloon_nrs)

    def find_duplicate_indices(arr, target):
        indices = []
        for i, num in enumerate(arr):
            if num == target:
                indices.append(i)
        return indices
    
    if duplicates_len > 0:
        indexes_to_keep = []
        for c, ind in enumerate(duplicates_ind_list):
            print("* BALLOON NR", str(ind), "IS ASSOCIATED MORE THAN ONCE")

            dupl_balloon_ind_list = (find_duplicate_indices(balloon_nrs, ind))

            # Get the detected characters for the same balloon
            duplicate_chars = [associated_chars[i] for i in dupl_balloon_ind_list]

            def distance(point, bbox):
                # Calculate the center of the bounding box
                bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # Calculate the Euclidean distance between the point and the bbox center
                return math.sqrt((point[0] - bbox_center[0]) ** 2 + (point[1] - bbox_center[1]) ** 2)
        
            # Initialize a list to store the indices
            indices_of_detected_chars = []

            # Find indices of elements of list A in list B
            for element in duplicate_chars:
                if element in detected_chars:
                    indices_of_detected_chars.append(detected_chars.index(element))

            char_boxes_to_calc = [character_bbox_list[i] for i in indices_of_detected_chars]

            tail_points_on_org_img = []
            for i in range(len(edges_list_all)):
                tail_points_on_org_img.append(edges_list_all[i][2])

            # Find the tail point that had more than one association
            tail_point_to_calc = tail_points_on_org_img[ind-1]

            # Calculate distances
            distance_to_bbox1 = distance(tail_point_to_calc, char_boxes_to_calc[0])
            distance_to_bbox2 = distance(tail_point_to_calc, char_boxes_to_calc[1])

            closer_char_ind = 0
            # Determine which bbox is closer
            if distance_to_bbox1 < distance_to_bbox2:
                closer_char_ind = 1
            elif distance_to_bbox1 > distance_to_bbox2:
                closer_char_ind = 2
            else:
                print("Both bounding boxes are equidistant from the point!!!")
            
            print("* THE CHARACTER THAT IS CLOSER TO THE BALLOON'S TAIL POINT IS SELECTED!")
            print(closer_char_ind)
            indexes_to_keep.append(c*2 + closer_char_ind - 1)

        def keep_elements_by_indexes(original_list, indexes_to_keep):
            return [elem for i, elem in enumerate(original_list) if i in indexes_to_keep]
        associated_chars = keep_elements_by_indexes(associated_chars, indexes_to_keep)
        balloon_nrs = keep_elements_by_indexes(balloon_nrs, indexes_to_keep)

        print("* REPLACEMENTS DONE!")

        correct_char_indexes = []
        character_present_or_not_list_new = ["NA"] * len(character_present_or_not_list)

        print(detected_chars)
        print(associated_chars)
        print(character_present_or_not_list)
        
        j = 0
        for i in range(len(character_present_or_not_list)): 
            if((i+1)%len(detected_chars) == 0):
                if "NA" not in character_present_or_not_list[i-(len(detected_chars)-1): i+(len(detected_chars)-1)]:
                    possible_chars = character_present_or_not_list[i-(len(detected_chars)-1): i+(len(detected_chars)-1)]
                    correct_char_indexes.append(i + possible_chars.index(associated_chars[j])) #detected_chars
                    j = j + 1
            if j > len(detected_chars):
                break;

        print(correct_char_indexes)
        print(character_present_or_not_list_new)
        for i, correct_ind in enumerate(correct_char_indexes):
            character_present_or_not_list_new[correct_ind-1] = associated_chars[i]
            
        character_present_or_not_list = character_present_or_not_list_new


    ## FINAL ASSOCIATIONS
    result_dict = {}
    for i in range(len(balloon_nrs)):
        #print("Balloon number", str(i+1), " -> ", associated_chars[i])
        result_dict[i] = associated_chars[i]
    '''

    texts_in_balloons = []
    for i in range(len(balloon_coordinates_list)):
        image_filename = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/cropped_image_" + str(i+1) + ".png"

        # reading the image
        img = cv2.imread(image_filename)

        # run OCR
        results = reader.readtext(img)
        
        if len(results) > 1:
            # REORDER results
            corner_y_coords = []
            corner_x_coords = []

            for res in results:
                corner_y_coords.append(int(res[0][0][1]))
                corner_x_coords.append(int(res[0][0][0]))

            groups, ordered_numbers = process_groups(corner_y_coords, corner_x_coords)

            new_order_list = []
            for group in ordered_numbers:
                for element in group:
                    new_order_list.append(element)

            new_order_indexes = find_locations(corner_y_coords, new_order_list)
            results = find_elements_by_indexes(results, new_order_indexes)
        
        # In the results list, some of the textboxes are often duplicates. We need to get rid of them
        unique_list = []
        seen = set()
        for sublist in results:
            # Convert the coordinates sublist to a tuple to make it hashable
            coordinates_tuple = tuple(map(tuple, sublist[0]))
            if coordinates_tuple not in seen:
                unique_list.append(sublist)
                seen.add(coordinates_tuple)
        results = unique_list

        # Store the detected text in a list
        detected_text_list = []

        # show the image and plot the results
        #plt.imshow(img)
        for idx, res in enumerate(results):
            # bbox coordinates of the detected text
            xy = res[0]
            #print(xy)
            xy1, xy2, xy3, xy4 = xy[0], xy[1], xy[2], xy[3]
            # text results and confidence of detection
            det, conf = res[1], res[2]

            # Save the detected text in the list
            detected_text_list.append((det, conf, xy))
            
            # show time :)
            plt.plot([xy1[0], xy2[0], xy3[0], xy4[0], xy1[0]], [xy1[1], xy2[1], xy3[1], xy4[1], xy1[1]], 'r-')
            plt.text(xy1[0], xy1[1], f'{det} [{round(conf, 2)}]')
            
            y = int(xy1[1])
            h = int(xy3[1] - xy1[1])
            x = int(xy1[0])
            w = int(xy2[0] - xy1[0])

            if y < 0: y = 0
            if x < 0: x = 0
            if h < 0: h = 0
            if w < 0: w = 0

            cropped_img = img[y:y+h, x:x+w]
 
            if (idx >= 10) & (cropped_img.size > 0):
                cv2.imwrite(f'/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/textboxes/balloon_{i}_text_box_{idx}.png', cropped_img)
            elif (len(cropped_img) == 0) | (cropped_img.size == 0):
                continue;
            else:
                cv2.imwrite(f'/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/textboxes/balloon_{i}_text_box_0{idx}.png', cropped_img)
        text_full = ""
        for detected in detected_text_list:
            text_full = text_full + detected[0] + " "
                
        texts_in_balloons.append(text_full)

        # Save the detected text to a file (optional)
        with open("detected_text.txt", "w") as file:
            for det, conf, xy in detected_text_list:
                file.write(f"{det} [{round(conf, 2)}] - Bounding Box: {xy}\n")

        # Show the plot
        #plt.savefig(f'/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/plot_{i}.png')
    '''
    for i in range(len(texts_in_balloons)):
        print("Balloon nr:", i+1)
        if len(result_dict)==0:
            print("Character: none found")
        else:
            print("Character:", result_dict[i])
        print("Text:", texts_in_balloons[i])
        print("\n")
    '''

    ## EĞER BALONDAKİ YAZI ALGILANMAZSA VE TEXTBOX OLARAK KAYDEDİLEMEYECEKSE; BALONUN KENDİSİ TEXTBOX OLARAK KAYDEDİLSİN
    ## BUARADA BALONLARIN SIRALAMASI SIKINTI YARATABİLİR!!!
    if "" in texts_in_balloons:
        un_ind = texts_in_balloons.index("")
        
        img_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/cropped_image_" + str(un_ind+1) + ".png"
        img = cv2.imread(img_path)
        cv2.imwrite(f'/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/textboxes/balloon_{un_ind}_text_box_00.png', img)
    
    ## EXCLAMATION & QUESTION MARK MODEL RUNS
    possible_exc_balloons = []

    for i in range(len(texts_in_balloons)):
        if (texts_in_balloons[i] == "") | (len(texts_in_balloons[i].strip()) == 1):
            possible_exc_balloons.append(i)

    ## PREDICT EXCLAMATION OR QUESTION MARK IN THE BALLOON
    detecteds = []
    for ind in possible_exc_balloons: 
        res = run_expression_model(expression_model, ind)
        
        exc_que_names = expression_model.names
        for r in res:
            for c in r.boxes.cls:
                detecteds.append(exc_que_names[int(c)])

    found_inds = [index for index, value in enumerate(detecteds) if value == 'question' or value == 'exclamation']

    def read_text_from_image(image_path):
        # Open the image file
        image = Image.open(image_path)

        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(image)

        return text

    for i in range(len(balloon_coordinates_list)):
        image_path = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/cropped_image_" + str(i+1) + ".png"
        result = read_text_from_image(image_path)

        #print("Text from speech balloon", str(i+1), ":", result.replace('\n', ' '))

    # Assuming your files are in the current directory
    directory = "/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/textboxes"

    # Get a list of all filenames in the directory
    files = os.listdir(directory)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    # Dictionary to store filenames grouped by balloon class
    file_groups = {}

    # Group filenames by balloon class
    for filename in files:
        parts = filename.split("_")
        balloon_class = parts[0] + "_" + parts[1]  # Extract the balloon class
        box_class = parts[3]  # Extract the box class
        if balloon_class not in file_groups:
            file_groups[balloon_class] = []
        file_groups[balloon_class].append(filename)
        
    # Sort filenames within each balloon class group
    for filenames in file_groups.values():
        filenames.sort()
            
    balloon_texts_list = []
    for balloon_class, filenames in sorted(file_groups.items()):
        
        text_trocr = ""
        for filename in sorted(filenames, key=lambda x: x.split("_")[3]):
            parts = filename.split("_")
            box_class = parts[4].split(".")[0]
            item = balloon_class + "_text_box_" + box_class + ".png"
            
            url = '/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/textboxes/' + item
            image = Image.open(url).convert("RGB")

            if image.size[1] <= 1:
                continue;

            pixel_values = trocr_processor(image, return_tensors="pt").pixel_values
            generated_ids = trocr_model.generate(pixel_values)

            text_trocr += trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0] + " "
            
        if len(text_trocr.strip()) <= 3: #Removing any punctuation if the text is short and possibly an expression
            text_trocr = re.sub(r'[^\w\s]','', text_trocr) 
                  
        if(len(text_trocr.strip()) <= 1):
            balloon_class_int = int(balloon_class[-1])

            if balloon_class_int >= len(found_inds):
                text_trocr = "UNKNOWN"
            else:
                found_inds_index = found_inds.index(balloon_class_int)
                text_trocr = detecteds[found_inds_index]
                
                if(text_trocr == "question"): text_trocr = "?"
                if(text_trocr == "exclamation"): text_trocr = "!"
                    
        balloon_texts_list.append(text_trocr)
        #print("TROCR:  Balloon nr.", str(int(balloon_class[-1])+1), "-->", text_trocr)

    associated_chars_final = []
    associated_balloons_final = []
    for i, element in enumerate(character_present_or_not_list):
        if element != "NA":
            which_balloon = int(i / len(detected_chars))
            which_char = i % len(detected_chars)

            associated_balloons_final.append(which_balloon)
            associated_chars_final.append(which_char)

    print(" ------------- END OF THE CODE -------------")
    print("BOOK", desired_book, "PANEL", desired_img)
    print("\nBALLOON TO PANEL RATIO:", balloon_to_panel_ratio)
    print("\nA. CHARACTER DETECTION\n")

    char_conf = []
    names = char_model.names
    detected_chars = []
    for r in character_result:
            for c in r.boxes.cls:
                detected_chars.append(names[int(c)])
    if len(detected_chars) > 0:
        for i in range(len(detected_chars)):
            print("\t",i+1,":", detected_chars[i], "\tconf:", round(float(character_result[0].boxes.conf[i]),3))
            char_conf.append(round(float(character_result[0].boxes.conf[i]),3))
    else:
        print("\t", "NO CHARACTER DETECTED")
        char_conf.append("NONE")
        
    print("\nB. SPEECH BALLOON DETECTION\n")
    print("\t", "Number of balloons:", len(balloon_texts_list))
    detected_balloon_nr = len(balloon_texts_list)
    print(balloon_texts_list)
    balloon_conf = []
    for i in range(len(balloon_texts_list)): #result_dict
        print("\t", "Balloon number",i+1, " -> conf:", round(float(balloon_result[0].boxes.conf[i]),3))
        balloon_conf.append(round(float(balloon_result[0].boxes.conf[i]),3))

    print("\nC. ALT BALLOON ASSOCIATIONS\n")
    for i in range(len(balloon_texts_list)):
         print("\t","Balloon number", str(i+1), " ->", associated_chars_dict[i])

    print("\nC. BALLOON ASSOCIATIONS\n")

    associated_balloon_nrs = []
    associated_chars_list = []
    for i in range(len(balloon_texts_list)):
        if i in associated_balloons_final:
            print("\t","Balloon number", str(i+1), " ->", detected_chars[associated_chars_final[associated_balloons_final.index(i)]])
            associated_chars_list.append(detected_chars[associated_chars_final[associated_balloons_final.index(i)]])
            associated_balloon_nrs.append(str(i+1))
        else:
            print("\t","Balloon number", str(i+1), " -> NaN")
            associated_chars_list.append("NaN") 

    print("\nD. ALT TEXT DETECTION\n")
    for i in range(len(balloon_texts_list)):
         print(associated_chars_dict[i], "->", balloon_texts_list[i].strip())    
    

    print("\nD. TEXT DETECTION\n")

    for i in range(len(balloon_texts_list)):
        if i in associated_balloons_final:
            print("\t", detected_chars[associated_chars_final[associated_balloons_final.index(i)]], ":\t", balloon_texts_list[i].strip())
        else:
            print("\t", "NaN:\t",  balloon_texts_list[i].strip())

    return detected_chars, char_conf, balloon_conf, detected_balloon_nr, associated_balloon_nrs, associated_chars_dict, balloon_texts_list, balloon_to_panel_ratio, panel_width, panel_length, panel_area