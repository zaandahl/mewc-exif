import os
import json
import pandas as pd
import piexif
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from lib_common import read_yaml
from lib_tools import process_detections,contains_animal
from PIL import Image

def print_exif(exif_dict):
    if 33437 in exif_dict["Exif"]:
        print(f'photo-fnumber-setting (detections): {exif_dict["Exif"][33437]}')
    if 34855 in exif_dict["Exif"]:
        print(f'photo-iso-setting (1st class): {exif_dict["Exif"][34855]}')
    if 33434 in exif_dict["Exif"]:
        print(f'photo-exposure-value (1st prob): {exif_dict["Exif"][33434]}')
    if 37386 in exif_dict["Exif"]:
        print(f'photo-focal-length (2nd class): {exif_dict["Exif"][37386]}')

config = read_yaml('config.yaml')
config['DEBUG'] = 0
for conf_key in config.keys():
    if conf_key in os.environ:
        config[conf_key] = os.environ[conf_key]

try:
    en_path = Path(config['INPUT_DIR'],config['EN_FILE'])
    en_out = pd.read_pickle(en_path)
except Exception as e:
    print(e)
    exit("ERROR: Unable to read en_out file.")

try:
    json_path = Path(config['INPUT_DIR'],config['MD_FILE'])
    with open(json_path, "r") as read_json:
        json_data = json.load(read_json)
except Exception as e:
    print(e)
    exit("ERROR: Unable to read MegaDetector json file")

print("Writing MD conf and orig date time exif data from " + str(len(json_data['images'])) + " images in " + config['MD_FILE'] + " to " + config['EN_CSV'])
en_out['date_time_orig'] = None
for json_image in tqdm(json_data['images']):
    if(contains_animal(json_image)):
        image_name = Path(json_image.get('file')).name
        image_stem = Path(json_image.get('file')).stem
        image_ext = Path(json_image.get('file')).suffix
        #input_path = Path(config['INPUT_DIR'],image_name)
        try:
            input_path = next(Path(config['INPUT_DIR']).rglob(image_name))
            img = Image.open(input_path)
            try:
                exif_dict = piexif.load(img.info["exif"])
            except:
                exif_dict = {'Exif': {}}
            try:
                en_out.loc[en_out['filename'].str.startswith(str(image_stem)) , 'date_time_orig'] = str(exif_dict["Exif"][36867].decode('UTF-8'))
            except:
                modified_time = os.path.getmtime(input_path)
                date_time_str = datetime.fromtimestamp(modified_time).strftime('%Y:%m:%d %H:%M:%S')
                en_out.loc[en_out['filename'].str.startswith(str(image_stem)), 'date_time_orig'] = date_time_str
            for idx in range(len(json_image['detections'])):
                en_out.loc[en_out['filename'].str.startswith(str(image_stem) + '-' + str(idx)), 'conf'] = json_image['detections'][idx]['conf']
        except Exception as e:
            print(e)
            print("ERROR: failed to process " + image_name)

try:
    en_out.to_pickle(Path(config["INPUT_DIR"],config["EN_FILE"]))
    en_out.to_csv(Path(config["INPUT_DIR"],config["EN_CSV"]))
except Exception as e:
    print(e)
    exit("ERROR: trouble writing output files")
# en_out = en_out.loc[((en_out.class_rank == 1.0) | (en_out.class_rank == 2.0)) & (en_out.filename.str.contains("-0\."))]
# en_out['filename'] = en_out['filename'].replace("-0\.", ".", regex = True)
# en_out = en_out.drop('label', axis = 1)

print("Writing metadata to " + str(len(json_data['images'])) + " images from " + config['MD_FILE'])
for json_image in tqdm(json_data['images']):
    if(contains_animal(json_image)):
        valid_image = process_detections(json_image,config['OVERLAP'],config['EDGE_DIST'],config['MIN_EDGES'],config['UPPER_CONF'],config['LOWER_CONF'])
        image_name = Path(json_image.get('file')).name
        image_stem = Path(json_image.get('file')).stem
        image_ext = Path(json_image.get('file')).suffix
        #input_path = Path(config['INPUT_DIR'],image_name)
        try:
            input_path = next(Path(config['INPUT_DIR']).rglob(image_name))
            detections = sum(valid_image)
            en_out.loc[en_out.filename == image_name, 'detections'] = detections # just housekeeping with the df
            img = Image.open(input_path)
            try:
                exif_dict = piexif.load(img.info["exif"])
            except:
                exif_dict = {'Exif': {}}
            if(41988 in exif_dict["Exif"] and type(exif_dict["Exif"][41988]) is int):
                exif_dict["Exif"][41988] = (exif_dict["Exif"][41988], 1) # hack to fix Bushnell data - this value needs to be a tuple
            if(int(config['DEBUG']) > 0):
                print(image_name)
                print("before")
                print_exif(exif_dict)
            if(int(config['DEBUG']) > 1):
                for key, value in exif_dict["Exif"].items():
                    print(key, value)
            # set the f_number 33437
            exif_dict["Exif"][33437] = (int(detections), 1)
            group_1 = en_out.loc[(en_out['filename'].str.startswith(str(image_stem))) & (en_out.class_rank == 1.0)]
            group_1 = group_1.reset_index()
            group_2 = en_out.loc[(en_out['filename'].str.startswith(str(image_stem))) & (en_out.class_rank == 2.0)]
            group_2 = group_2.reset_index()
            if(len(group_1) > 0):
                class_1 = group_1.loc[group_1['conf'].idxmax(), 'class_id']
                prob_1 = group_1.loc[group_1['conf'].idxmax(), 'prob']
            #class_1 = en_out.loc[(en_out['filename'].str.contains(str(image_stem))) & (en_out.class_rank == 1.0), 'class_id']
            #prob_1 = en_out.loc[(en_out['filename'].str.contains(str(image_stem))) & (en_out.class_rank == 1.0), 'prob']
            #class_2 = en_out.loc[(en_out['filename'].str.contains(str(image_stem))) & (en_out.class_rank == 2.0), 'class_id']
            #if(len(class_1) > 0): 
                # set iso 34855
                #exif_dict["Exif"][34855] = int(class_1.values[0])
                exif_dict["Exif"][34855] = int(class_1)
                # set exposure time 33434
                #exif_dict["Exif"][33434] = (int(round(prob_1.values[0]*8640,0)), 8640)
                #exif_dict["Exif"][33434] = (min(int(round(prob_1.values[0]*100,0)),99),1)
                exif_dict["Exif"][33434] = (min(int(round(prob_1*100,0)),99),1)
            else: # clean up if there are no efficientnet values to store
                exif_dict["Exif"].pop(34855, None)
                exif_dict["Exif"].pop(33434, None)
            if(len(group_2) > 0):
                class_2 = group_2.loc[group_2['conf'].idxmax(), 'class_id']
                # set focal length 37386
                #exif_dict["Exif"][37386] = (int(class_2.values[0]), 1)
                exif_dict["Exif"][37386] = (int(class_2), 1)
            else: # clean up if there are no efficientnet values to store
                exif_dict["Exif"].pop(37386, None)
            if(int(config['DEBUG']) > 0):
                print("After")
                print_exif(exif_dict)
            exif_bytes = piexif.dump(exif_dict)
            img.save(input_path, exif=exif_bytes)
        except Exception as e:
            print(e)
            print("ERROR: failed to process " + image_name)