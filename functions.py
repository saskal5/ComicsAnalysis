## FUNCTIONS

import os
import numpy as np
import re
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

def sort_key(filename):
        # Extract the numerical part from the filename
        match = re.match(r'^ROI_(\d+)\.png$', filename)
        if match:
            # Return a tuple with the number of digits after underscore and the numerical part
            return len(match.group(1)), int(match.group(1))
        else:
            return filename
        
def delete_files_in_folder(folder_path):
        # Get the list of files in the folder
        files = os.listdir(folder_path)
        
        # Iterate over each file and delete it
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

def crop_and_save(character_result, balloon_result, img, img_gray):
    ## STORE THE COORDINATES OF THE CHARACTERS
    character_coords_cropped = []
    ## CROPPING IMAGES BASED ON SPEECH BALLOONS
    balloon_coordinates_list = []

    cropped_img_count = 0
    for box in character_result[0].boxes.xyxy: 
        cropped_img_count += 1
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        coords = (xA, yA, xB, yB)
        img_cropped = img.crop(coords)
        character_coords_cropped.append(coords)
        img_name = "img_cropped_char" + str(cropped_img_count) + ".png"
        img_cropped.save('./output_folder/images_cropped/cropped/' + img_name)
        
    cropped_img_count = 0
    for box in balloon_result:  #balloon_result[0].boxes.xyxy
        cropped_img_count += 1
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        coords = (xA, yA, xB, yB)
        img_cropped = img_gray.crop(coords)
        balloon_coordinates_list.append(coords)
        img_name = "img_cropped_" + str(cropped_img_count) + ".png"
        img_cropped.save('./output_folder/images_cropped/cropped/' + img_name)
        
    return character_coords_cropped, balloon_coordinates_list

## DRAW THE BOUNDING BOXES OF CHARACTERS
def draw_boxes(image, boxes):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        color = (255, 0, 0)  # Red color for all boxes
        cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), color, 2)
        cv2.putText(image, str(i + 1), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)

## PREPARE THE PANEL BEFORE THE USE OF MODEL (MASKING)
def mask_single_image(input_image_name, input_folder, output_folder, threshold=220):
    """
    Apply a mask to a grayscaled input image and save the masked image.

    Parameters:
    - input_image_name (str): Name of the input image (including the extension).
    - input_folder (str): Path to the folder containing the input image.
    - output_folder (str): Path to save the masked image.
    - threshold (int): Threshold value for the mask (default is 128).
    """

    # Construct the full path to the input image
    input_image_path = os.path.join(input_folder, input_image_name)

    # Load the input image and convert to grayscale
    img = Image.open(input_image_path).convert('L')

    # Construct the full path to the mask image based on the input image name
    base_name, ext = os.path.splitext(input_image_name)
    mask_filename = os.path.join(input_folder, base_name + ext)

    # Check if the mask file exists
    if os.path.exists(mask_filename):
        # Load the mask
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        # Apply the threshold to the mask
        _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        # Apply the binary mask to the grayscale image
        img_array = np.array(img)
        img_array[binary_mask == 0] = 0
        masked_img = Image.fromarray(img_array)

        # Save the masked image to the output path
        output_image_path = os.path.join(output_folder, input_image_name)
        masked_img.save(output_image_path)
        print(f"Masked image saved to: {output_image_path}")
    else:
        print(f"Mask not found for image: {input_image_name}")
        
## SHOW THE SPEECH BALLOON CONTOURS WITH RED        
def overlay_largest_area_contours_with_numbers(original_image_path, cropped_images_folder, coordinates_list, threshold, output_path):
    # Read the original image
    original_image = cv2.imread(original_image_path)

    org_img_area = original_image.shape[0] * original_image.shape[1]

    # Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the original image at {original_image_path}")
        return None

    # Get a list of all files in the cropped images folder
    cropped_image_files = [f for f in os.listdir(cropped_images_folder) if os.path.isfile(os.path.join(cropped_images_folder, f)) and f.startswith("img_cropped_")]

    # Sort the list of cropped image files
    cropped_image_files.sort()

    # List to store the contours with the largest area
    largest_area_contours_list = []

    # Iterate through each cropped image and its coordinates
    for cropped_image_file, coordinates in zip(cropped_image_files, coordinates_list):
        # Read the cropped image
        cropped_image_path = os.path.join(cropped_images_folder, cropped_image_file)
        cropped_image = cv2.imread(cropped_image_path, cv2.IMREAD_UNCHANGED)

        # Check if the image is loaded successfully
        if cropped_image is None:
            print(f"Error: Unable to load the cropped image at {cropped_image_path}")
            continue

        # Get the coordinates of the cropped part
        x_cropped, y_cropped, w_cropped, h_cropped = coordinates

        # Convert the coordinates to integer values
        x_cropped, y_cropped, w_cropped, h_cropped = int(x_cropped), int(y_cropped), int(w_cropped), int(h_cropped)

        # Threshold the cropped image to create a binary mask based on the whitest pixels
        _, binary_mask = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_height_threshold = int(h_cropped / 8)
        contour_width_threshold = int(h_cropped / 8)

        #print("CONTOUR!!!")
        #print(h_cropped)
        #for contour in contours:
        #    print(cv2.boundingRect(contour)[3], contour_height_threshold)
        #    print(cv2.boundingRect(contour)[2], contour_width_threshold)


        ### HEIGHT CRITERIA!!!
        # Filter contours based on height and width
        filtered_contours = [contour for contour in contours if cv2.boundingRect(contour)[3] > contour_height_threshold]
        filtered_contours = [contour for contour in filtered_contours if cv2.boundingRect(contour)[2] > contour_width_threshold]
        filtered_contours = [contour for contour in filtered_contours if cv2.contourArea(contour) > 1000]
        #print("FILTERED", filtered_contours)


        # Find index of contour with largest area from filtered contours
        if filtered_contours:
            #largest_width_contour_index = np.argmax([cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] for contour in filtered_contours])
            #largest_width_contour_index = np.argmax([cv2.contourArea(contour) for contour in filtered_contours])

            # Find the index of the contour with the largest width and largest height
            largest_width_contour_index = np.argmax([cv2.boundingRect(contour)[2] for contour in filtered_contours])

            # Get the contour with the largest width and largest height from the list
            largest_width_contour = filtered_contours[largest_width_contour_index] # contours -> filtered_contours
            print("LARGEST")
            print(cv2.boundingRect(largest_width_contour)[3])
                
            largest_area_contour = largest_width_contour
            print("AMANIN", cv2.boundingRect(largest_area_contour)[2])
            print("AMANIN", cv2.boundingRect(largest_area_contour)[3])

            # Map the contour coordinates to the original image coordinates
            largest_area_contour_mapped = largest_area_contour + np.array([x_cropped, y_cropped])

            # Add the contour with the largest area to the list
            largest_area_contours_list.append(largest_area_contour_mapped)   
        else:
            return [], 0, 0, 0, 0

    
    # Draw all the contours with the largest area on the original image
    cv2.drawContours(original_image, largest_area_contours_list, -1, (0, 0, 255), thickness=cv2.FILLED)

    total_contour_area = 0
    # Add numbers to each contour
    for i, contour in enumerate(largest_area_contours_list, 1):
        # Calculate the centroid of the contour
        total_contour_area += cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Put the number on the original image
            cv2.putText(original_image, str(i), (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)

    balloon_to_panel_ratio = round((total_contour_area / org_img_area), 3)
    

    # Save the result image
    cv2.imwrite(output_path, original_image)

    print(f"Result image with numbers saved to: {output_path}")

    return largest_area_contours_list, balloon_to_panel_ratio, original_image.shape[0], original_image.shape[1], original_image.shape[0] * original_image.shape[1]

## CROP THE ORIGINAL PANEL FOR EACH TIME A SPEECH BALLON DETECTED BASED ON THE CONTOUR
def crop_original_image(original_image_path, largest_area_contours_list, output_folder="cropped_images"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the original image
    original_image = cv2.imread(original_image_path)

    # Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the original image at {original_image_path}")
        return None

    cropped_balloon_from_org = []
    # Iterate through each contour with the largest width
    for i, contour in enumerate(largest_area_contours_list, 1):
        # Create a mask for the current contour
        mask = np.zeros_like(original_image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Extract the region of interest (ROI) using the mask
        cropped_image = cv2.bitwise_and(original_image, mask)

        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        cropped_balloon_from_org.append((x,y,w,h))

        # Crop the result image based on the bounding box
        cropped_result = cropped_image[y:y+h, x:x+w]

        # Save the cropped result image
        output_path = os.path.join(output_folder, f"cropped_image_{i}.png")
        cv2.imwrite(output_path, cropped_result)

        print(f"Cropped image with contour and bounding box {i} saved to: {output_path}")

    return cropped_balloon_from_org

## FIND CORNERS OF THE SPEECH BALLOON CONTOUR
def find_contour_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        #print("EPSILON", epsilon)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        return corners
    else:
        return None

## SAVE THE IMAGE OF CONTOUR WITH CORNERS
def save_corners(image, corners, output_path):
    image_with_corners = image.copy()
    cv2.drawContours(image_with_corners, [corners], -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, image_with_corners)

## SAVE THE IMAGE OF DIFFERENCE REGION (BETWEEN ORIGINAL CORNERS AND THE BLURRED CORNERS)   
def save_difference_region(original_image, blurred_image, output_path):
    original_corners = find_contour_corners(original_image)
    blurred_corners = find_contour_corners(blurred_image)

    if original_corners is not None and blurred_corners is not None:
        original_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(original_mask, [original_corners], -1, 255, thickness=cv2.FILLED)

        blurred_mask = np.zeros(blurred_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(blurred_mask, [blurred_corners], -1, 255, thickness=cv2.FILLED)

        difference_mask = cv2.subtract(original_mask, blurred_mask)
        result = cv2.bitwise_and(original_image, original_image, mask=difference_mask)

        cv2.imwrite(output_path, result)
    else:
        print("Error: Contours not found in one or both images.")

## FIND THE TAIL OF THE SPEECH BALLOON
def find_tail_point(original_corners, blurred_contour):
    max_distance = 0
    tail_point = None

    for corner in original_corners:
        x, y = corner[0]
        distance = np.inf

        for contour_point in blurred_contour:
            distance = min(distance, np.linalg.norm(np.array([x, y]) - contour_point[0]))

        if distance > max_distance:
            max_distance = distance
            tail_point = (int(x), int(y))

    return tail_point

## SAVE THE IMAGE WITH THE TAIL OF THE SPEECH BALLOON
def mark_tail_point(original_image, tail_point, output_path):
    image_with_marker = original_image.copy()
    cv2.circle(image_with_marker, tail_point, 5, (0, 165, 255), -1)  # Orange circle
    cv2.imwrite(output_path, image_with_marker)
    
## FIND CONTOUR WITH CORNERS   
def find_contour_with_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        return contour, corners
    else:
        return None, None

## SAVE CONTOUR WITH CORNERS 
def save_contour_with_corners(image, contour, corners, output_path):
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, [contour], -1, (0, 255, 0), 2)
    
    for corner in corners:
        cv2.circle(image_with_contour, tuple(corner[0]), 5, (0, 0, 255), -1)  # Red circle

    cv2.imwrite(output_path, image_with_contour)

## FIND SMALLEST ANGLE IN THE TRIANGLE 
def find_smallest_angle_triangle(corners, tail_point):
    
    # Find the index where the second array is present in the original array
    index = np.where(np.all(corners == tail_point, axis=(1, 2)))[0][0]

    # Extract the adjacent elements
    filtered_elements = [corners[(index - 1) % len(corners)],
                     corners[index],
                     corners[(index + 1) % len(corners)]]

    
    corners_filtered = []
    # Print the filtered elements
    for element in filtered_elements:
        corners_filtered.append(element)
    
    smallest_angle = np.inf
    best_triangle = None

    for i in range(len(corners)):
        # Get three adjacent corner points
        corner1 = corners[i][0]
        corner2 = corners[(i + 1) % len(corners)][0]
        corner3 = corners[(i + 2) % len(corners)][0]

        # Calculate vectors
        vector1 = corner1 - corner2
        vector2 = corner3 - corner2

        # Calculate dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        # Calculate angle in radians
        angle_radians = np.arccos(dot_product / magnitude_product)

        # Convert angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Update smallest angle
        if angle_degrees < smallest_angle:
            smallest_angle = angle_degrees
            best_triangle = (corner1, corner2, corner3)

    return best_triangle

def mark_triangle(image, triangle, output_path):
    image_with_triangle = image.copy()
    
    for corner in triangle:
        cv2.circle(image_with_triangle, tuple(corner), 3, (255, 0, 0), -1)  # Blue circle

    cv2.line(image_with_triangle, tuple(triangle[0]), tuple(triangle[1]), (255, 0, 0), 2)
    cv2.line(image_with_triangle, tuple(triangle[1]), tuple(triangle[2]), (255, 0, 0), 2)
    cv2.line(image_with_triangle, tuple(triangle[2]), tuple(triangle[0]), (255, 0, 0), 2)

    cv2.imwrite(output_path, image_with_triangle)


def visualize_and_extend(original_image, contour_coordinates, smallest_angle_triangle, output_path):
    image_with_visualization = original_image.copy()

    # Offset the coordinates based on the contour of the cropped image in the original image
    contour_offset = (contour_coordinates[0], contour_coordinates[1])
    smallest_angle_triangle_offset = [(point[0] + contour_offset[0], point[1] + contour_offset[1]) for point in smallest_angle_triangle]

    # Calculate the angles within the triangle
    angles = []
    corners = []
    for i in range(len(smallest_angle_triangle_offset)):
        corner1 = smallest_angle_triangle_offset[i]
        corner2 = smallest_angle_triangle_offset[(i + 1) % len(smallest_angle_triangle_offset)]
        corner3 = smallest_angle_triangle_offset[(i + 2) % len(smallest_angle_triangle_offset)]

        vector1 = np.array(corner1) - np.array(corner2)
        vector2 = np.array(corner1) - np.array(corner3)

        # Calculate the angle between two vectors
        angle = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
        angles.append(angle)
        corners.append(corner1)

    def reorder_elements(lst):
        # Find the index of the smallest element
        min_index = lst.index(min(lst))
    
        # Reorder the elements such that the smallest element stays in the middle
        reordered_lst = [lst[(min_index + 1) % 3], min(lst), lst[(min_index + 2) % 3]]
    
        return reordered_lst
    
    angles = reorder_elements(angles)

    # Find the index of the corner with the smallest angle
    smallest_angle_index = np.argmin(angles)


    # Find the corner with the smallest angle
    smallest_angle_corner = smallest_angle_triangle_offset[smallest_angle_index]
    #print("smallest_angle_corner", smallest_angle_corner)
    
    other_corner_index1 = (smallest_angle_index+1) %len(smallest_angle_triangle_offset)
    other_corner_index2 = (smallest_angle_index+2) %len(smallest_angle_triangle_offset)
    
    # Find the two edges sharing the corner with the smallest angle
    edge1_start, edge1_end = smallest_angle_triangle_offset[other_corner_index1], smallest_angle_triangle_offset[smallest_angle_index]
    edge2_start, edge2_end = smallest_angle_triangle_offset[other_corner_index2], smallest_angle_triangle_offset[smallest_angle_index]

    #print("EDGE STARTS")
    #print(edge1_start, edge2_start)

    edge1_start_arr = np.array(edge1_start)
    edge2_start_arr = np.array(edge2_start)
    smallest_angle_corner_arr = np.array(smallest_angle_corner)
    
    ba = edge1_start_arr - smallest_angle_corner_arr
    bc = edge2_start_arr - smallest_angle_corner_arr

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    #print("ANGLE", np.degrees(angle))
    # If the extending edges create less than 45 degrees, then do this: otherwise not needed.
    if np.degrees(angle) < 45:
        # Edge starting points are made even further from each other to enlarge the extension area to correctly insersect detected characters.
        if edge1_start[0] > edge2_start[0]:  # Check if first element of A is bigger than B
            if edge1_start[1] > edge2_start[1]:  # Check if second element of A is bigger than B
                edge1_start = (edge1_start[0] + 5, edge1_start[1] + 5)
                edge2_start = (edge2_start[0] - 5, edge2_start[1] - 5)
            else:
                edge1_start = (edge1_start[0] + 5, edge1_start[1] - 5)
                edge2_start = (edge2_start[0] - 5, edge2_start[1] + 5)
        else:
            if edge1_start[1] > edge2_start[1]:  # Check if second element of A is bigger than B
                edge1_start = (edge1_start[0] - 5, edge1_start[1] + 5)
                edge2_start = (edge2_start[0] + 5, edge2_start[1] - 5)
            else:
                edge1_start = (edge1_start[0] - 5, edge1_start[1] - 5)
                edge2_start = (edge2_start[0] + 5, edge2_start[1] + 5)

    # Extend both adjacent edges in the direction of the corner point with the smallest angle
    end_point_extended1 = extend_line(image_with_visualization, edge1_start, edge1_end, smallest_angle_corner)
    end_point_extended2 = extend_line(image_with_visualization, edge2_start, edge2_end, smallest_angle_corner)
    
    edges_list = [end_point_extended1, end_point_extended2, edge1_end]

    # Draw the triangle on the original image
    cv2.polylines(image_with_visualization, [np.array(smallest_angle_triangle_offset)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw the corner point with the smallest angle
    cv2.circle(image_with_visualization, tuple(map(int, smallest_angle_corner)), 5, (0, 255, 0), -1)

    cv2.imwrite(output_path, image_with_visualization)
    return edges_list

def extend_line(image, start_point, end_point, target_point):
    # Extend the line from start_point to target_point
    direction_vector = np.array(target_point) - np.array(start_point)
    scale_factor = max(image.shape[0], image.shape[1])  # Scale factor for line length

    # Extend the edge
    end_point_extended = end_point + scale_factor * direction_vector
    cv2.line(image, tuple(map(int, end_point)), tuple(map(int, end_point_extended)), (0, 255, 255), 2)  # Yellow color for extended lines
    return end_point_extended

def is_character_intersecting_region(character_bbox, extended_lines_polygon):
    x, y, width, height = character_bbox
    character_polygon = Polygon([(width, y), (width, height), (x, height), (x, y)])
    return character_polygon.intersects(extended_lines_polygon)

def find_locations(original_list, subset):
    locations = [original_list.index(num) for num in subset]
    return locations

def find_elements_by_indexes(another_list, indexes):
    elements = [another_list[index] for index in indexes]
    return elements

def process_groups(numbers, partners):

    temp = numbers.copy() 
    temp.sort()
    
    # Group numbers
    groups = []
    current_group = []
    for i in range(len(temp) - 1):
        current_group.append(temp[i])
        if temp[i+1] - temp[i] > 10:
            groups.append(current_group)
            current_group = []
    current_group.append(temp[-1])
    groups.append(current_group)

    # Create a new list ordered according to the second list
    ordered_numbers = []
    for group in groups:
        locs1 = find_locations(numbers, group)
        partner_numbers = find_elements_by_indexes(partners, locs1)
        partner_numbers.sort()
        locs2 = find_locations(partners, partner_numbers)
        ordered_numbers.append(find_elements_by_indexes(numbers, locs2))
    
    return groups, ordered_numbers
