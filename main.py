import sys
import streamlit as st
import pydicom
import pandas as pd
import numpy as np
from datetime import datetime
import os
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from matplotlib.patches import Circle
from ImageRetrievalSystem import ImageRetrievalSystem
import time
import math


def compute_match_percentage(initial_description, additional_description):
    matcher = SequenceMatcher(None, initial_description, additional_description)
    match_ratio = matcher.ratio() * 100
    return match_ratio


def compute_date_closeness(initial_date, additional_date):
    initial_date = datetime.strptime(initial_date, "%Y-%m-%d")
    additional_date = datetime.strptime(additional_date, "%Y-%m-%d")
    days_difference = abs((initial_date - additional_date).days)
    return days_difference


def extract_shape_features(pixel_array):
    # Binarize the image based on a threshold (adjust threshold as needed)
    threshold = 200
    binary_image = np.where(pixel_array > threshold, 1, 0)

    # Label connected components
    labeled_image = label(binary_image)

    # Extract shape features from labeled regions
    regions = regionprops(labeled_image)

    # Plot the image with highlighted shapes
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the initial DICOM image
    ax[0].imshow(pixel_array, cmap=plt.cm.gray)
    ax[0].set_title('Initial Image')

    # Display the image with highlighted shapes
    ax[1].imshow(pixel_array, cmap=plt.cm.gray)
    ax[1].set_title('Image with Highlighted Shapes')

    for region in regions:
        # Get the centroid and equivalent diameter of the region
        centroid = region.centroid
        diameter = region.equivalent_diameter

        # Highlight the shape by drawing a circle
        circle = Circle(centroid[::-1], diameter / 2, fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(circle)

    # Convert the plot to an image
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())

    return image


def extract_context_features(dicom_data):
    modality = dicom_data.Modality
    body_part = dicom_data.BodyPartExamined
    study_description = dicom_data.StudyDescription
    study_date = dicom_data.StudyDate
    study_date = datetime.strptime(study_date, "%Y%m%d").date().strftime("%Y-%m-%d")
    image_type = dicom_data.ImageType
    image_type = [str(val) for val in image_type]
    return modality, body_part, study_description, study_date, image_type


def display_pixel_data_and_tags(dicom_files):
    print('dci', dicom_files)
    num_files = len(dicom_files)
    for i, file in enumerate(dicom_files):
        print('file', file)
        # Read the DICOM file
        dicom_data = pydicom.dcmread(file)

        # Get the pixel data
        pixel_data = dicom_data.pixel_array

        # Create a new column for each pair of images
        col1, col2 = st.columns(2)

        # Display the image in the respective column
        modality, body_part, study_description, study_date, image_type = extract_context_features(
            dicom_data)

        data = [
            {'Tag': 'Modality', 'Value': modality},
            {'Tag': 'Body Part Examined', 'Value': body_part},
            {'Tag': 'Study Description', 'Value': study_description},
            {'Tag': 'Study Date', 'Value': study_date},
            {'Tag': 'image_type', 'Value': str(image_type)}
        ]
        df = pd.DataFrame(data)
        st.write(f"Result #{i}")
        with col1:
            st.subheader("DICOM Image")
            fig, ax = plt.subplots()
            ax.imshow(pixel_data, cmap=plt.cm.gray)
            ax.axis('off')
            col1.pyplot(fig)
        with col2:
            st.subheader("DICOM Tags")
            st.dataframe(df.set_index(df.columns[0]), use_container_width=True, height=580)

        # If there is an odd number of images, add an empty column for alignment
    if num_files % 2 != 0:
        st.empty()


def extract_and_compare_features(initial_data):
    additional_directory = "./assets/poll"
    list_of_files = []
    comparison_data = []
    for file_name in os.listdir(additional_directory):
        if initial_data['File'] != file_name:
            modality, body_part, study_description, study_date, image_type = extract_context_features(
                pydicom.dcmread('./assets/poll/' + file_name))
            computed_row = {}
            comparison_row = {
                'File': file_name,
                'Modality': modality,
                'Body Part Examined': body_part,
                'Study Description': study_description,
                'Study Date': study_date,
                'Image Type': image_type
            }
            if (
                    initial_data['Modality'] == comparison_row['Modality']
                    and initial_data['Body Part Examined'] == comparison_row['Body Part Examined']
            ):
                computed_row['File'] = file_name
                for key, value in initial_data.items():
                    if key == 'Image Type':
                        match_percentage = compute_match_percentage(value, comparison_row[key])
                        computed_row["Image Type Match (%)"] = match_percentage
                    elif key == 'Study Date':
                        closeness = compute_date_closeness(value, comparison_row[key])
                        computed_row['Study Date Closeness (days)'] = closeness
                    elif key == 'Study Description':
                        match_percentage = compute_match_percentage(value, comparison_row[key])
                        computed_row['Study Description Match (%)'] = match_percentage
                    else:
                        computed_row[value] = comparison_row[key]
                comparison_data.append(computed_row)
                list_of_files.append(additional_directory + '/' + file_name)

    sorted_comparison_data = sorted(
        comparison_data,
        key=lambda x: (
            x['Study Date Closeness (days)'],
            x['Image Type Match (%)'],
            x['Study Description Match (%)']
        ),
        reverse=False
    )

    return pd.DataFrame(sorted_comparison_data), list_of_files


def display_dicom_pixel_data(ds):
    # Extract pixel data
    pixel_data = ds.pixel_array

    # Display the pixel data using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(pixel_data, cmap=plt.cm.gray)
    ax.axis('off')

    # Convert the matplotlib figure to a Streamlit image
    st.pyplot(fig)

    return pixel_data


def extract_dicom_tags(dicom_data):
    data = []
    for tag in dicom_data.keys():
        tag_name = pydicom.datadict.keyword_for_tag(tag)
        data.append({'Tag': tag_name, 'Value': str(dicom_data[tag].value)})
    df = pd.DataFrame(data)
    return df


def open_dicom_file(file):
    ds = pydicom.dcmread(file)

    return ds


class DICOMFeatureExtractor:
    def __init__(self):
        self.dicom_files = []
        self.dicom_dir = 'assets/poll'
        self.comparison_df = []

    def run(self):
        st.set_page_config(
            page_title="Home | Content and Context Based Feature Extraction", layout='wide')
        # Create sidebar
        st.sidebar.title("Content and Context Based Feature Extraction")

        pages = st.sidebar.radio("Pages", ("Home", "Repository"))
        home_button = pages == "Home"
        repository_button = pages == "Repository"

        if home_button:
            self.home_page()
        else:
            self.repository_page()

        st.sidebar.header("About")
        st.sidebar.info("The DICOM files are used to compare features extracted from other DICOM images.")

    def home_page(self):
        st.title("Extract Feature from File")
        st.write("Please upload the query image.")
        query_image = st.file_uploader("Upload Query Image", type=["dcm", "dicom"])
        if query_image is not None:
            st.success("Upload complete")
            dicom_data = open_dicom_file(query_image)
            df_tags = extract_dicom_tags(dicom_data)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("DICOM Tags")
                st.dataframe(df_tags.set_index(df_tags.columns[0]), use_container_width=True, height=580)
            with col2:
                st.subheader("DICOM Image")
                pixel_data = display_dicom_pixel_data(dicom_data)
            if st.button("Extract Features"):
                # Process the query image
                start_time = time.time()
                image = extract_shape_features(pixel_data)

                st.image(image, caption='Image with Highlighted Shapes')

                modality, body_part, study_description, study_date, image_type = extract_context_features(
                    dicom_data)
                end_time = time.time()
                extraction_time = end_time - start_time
                initial_data = [{
                    'Modality': modality,
                    'Body Part Examined': body_part,
                    'Study Description': study_description,
                    'Study Date': study_date,
                    'Image Type': image_type
                }]

                df = pd.DataFrame(initial_data)

                st.table(df)
                st.success(f"Extraction Time: {extraction_time:.4f} seconds")
                if st.button('Store Features'):
                    print('features')

    def repository_page(self):
        # Read DICOM files from the directory
        for file in sorted(os.listdir(self.dicom_dir)):
            if file.endswith('.dcm'):
                self.dicom_files.append(file)

        st.write("## DICOM Files")

        st.session_state.more_stuff = False
        n_rows = 1 + len(self.dicom_files) // 3
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(4) for r in rows]
        cols = [column for row in cols_per_row for column in row]

        for image_index, image in enumerate(self.dicom_files):
            cols[image_index].image('./assets/image/icon.png', width=100)
            button_label = f"View Details {image}"
            cols[image_index].text(image)

            if cols[image_index].button(button_label):
                data = []
                start_time = time.time()
                st.success("Extraction complete")
                st.write(f" ## Extracted context features from file: {image}")
                modality, body_part, study_description, study_date, image_type = extract_context_features(
                    pydicom.dcmread('./assets/poll/' + image))
                initial_data = {
                    'File': image,
                    'Modality': modality,
                    'Body Part Examined': body_part,
                    'Study Description': study_description,
                    'Study Date': study_date,
                    'Image Type': image_type
                }

                data.append({
                    'File': image,
                    'Modality': modality,
                    'Body Part Examined': body_part,
                    'Study Description': study_description,
                    'Study Date': study_date,
                    'Image Type': image_type
                })
                df = pd.DataFrame(data)
                st.table(df)

                self.comparison_df, list_of_files = extract_and_compare_features(initial_data)

                ds = pydicom.dcmread('./assets/poll/' + image)

                # Extract the pixel data
                pixel_data = ds.pixel_array

                col1, col2 = st.columns(2)
                with col1:
                    st.write("## Query Image")
                    display_dicom_pixel_data(pydicom.dcmread('./assets/poll/' + image))
                st.write("## DICOM Metadata Comparison")
                st.table(self.comparison_df)
                image_retrieval_system = ImageRetrievalSystem(list_of_files)
                image_retrieval_system.load_images()
                # Query the retrieval system
                top_ranked_images = image_retrieval_system.query_image(pixel_data)
                end_time = time.time()
                comparison_time = end_time - start_time
                st.success(f"Comparison: {comparison_time:.4f} seconds")
                st.write("## Refined search by Content")

                print('top_ranked_images', top_ranked_images)
                display_pixel_data_and_tags(top_ranked_images)


if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "main.py"]
    feature_extractor = DICOMFeatureExtractor()
    feature_extractor.run()
