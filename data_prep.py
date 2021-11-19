import os
import numpy as np
import pandas as pd
from skimage import io

def load_images():
    img_names = os.listdir('./Images')
    data_dir = os.path.join(os.getcwd(), 'Images')
    X = []
    for img in img_names:
        img_path = os.path.join(data_dir, img)
        X.append(io.imread(img_path))
    
    print(f'Loaded {len(X)} images...')
    
    return np.asarray(X), img_names

def load_annotations():
    annotation_labels= os.listdir('./Annotations')
    data_dir = os.path.join(os.getcwd(), 'Annotations')
    data_frames = []
    for annotation in annotation_labels:
        annotation_path = os.path.join(data_dir, annotation, 'annotations.csv')
        csv = pd.read_csv(annotation_path)
        csv = csv.rename(columns={csv.columns[0]: "Image"})
        data_frames.append(csv)
    
    data_frame = data_frames[0]
    for i in range(1, len(data_frames)):
        data_frame = pd.merge(data_frame, data_frames[i], how='outer')
    
    print(f'The annotations reference {data_frame.shape[0]} images...')

    return data_frame, annotation_labels

def load_cleaned_data():
    """There are images listed in the annotations that don't exist.
    So let's get rid of them and return a clean dataset."""
    X, img_names = load_images()
    annotations, annotation_labels = load_annotations()
    annotations = annotations.dropna()
    mask = np.isin(img_names, annotations['Image'])

    print(f'We are deleting {len(mask) - np.sum(mask)} images...')
    print(f'Leaving a total of {np.sum(mask)} images...')

    return X[mask], np.asarray(annotations[annotation_labels]), annotation_labels

if __name__ == '__main__':
    images, annotations, annotation_labels = load_cleaned_data()
    print(images)
    print(annotations)
    print(annotation_labels)