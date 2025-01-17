from ultralytics import YOLO
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

char_model_conf = 0.7
balloon_model_conf = 0.6
exp_model_conf = 0.2

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def load_trocr_processor():
    return trocr_processor

def load_trocr_model():
    return trocr_model

### CHARACTER DETECTION

def load_char_model():
    # Your model loading code
    model = YOLO('./models/YOLOv8_V2_best.pt')
    return model

def run_char_model(model, book, image):
    # New version model is trained on grayscale images. So, convertion was needed.
    #img = Image.open('/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/' + image).convert('L')
    #img.save('/Users/azatsaskal/Documents/UGent/Masters Thesis/speech_balloons_hq_masked/test_random_panels/images_cropped/original/gray_' + image)
    
    result = model('/Users/azatsaskal/Documents/UGent/Masters Thesis/PDFs_HQ/' + book + "/" + image, 
                                        conf=char_model_conf, 
                                        save=False,
                                        project= './output_folder/images_cropped/')
    return result

### SPEECH BALLOON DETECTION

def load_balloon_model():
    # Your model loading code
    model = YOLO('./models/YOLOv8_balloon_best.pt')
    return model

def run_balloon_model(model, image):
    # Your model loading code
    result = model('./output_folder/' + image, 
                                           conf=balloon_model_conf, 
                                           save=True)
    return result

### EXPRESSION DETECTION

def load_expression_model():
    model = YOLO('./models/YOLOv8_expression_best.pt.pt')
    return model

def run_expression_model(model, ind):
    result = model("./output_folder/images_cropped/original/cropped_image_" + str(ind+1) + ".png", 
                                           conf=exp_model_conf, 
                                           save=False)
    return result
