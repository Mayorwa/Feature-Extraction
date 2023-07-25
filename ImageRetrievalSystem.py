import os
import numpy as np
import pydicom
from skimage import feature, measure
from collections import defaultdict
from PIL import Image, ImageDraw


def highlight_features(image, features):
    threshold = 127
    binary_image = image > threshold

    # Find contours in the binary image
    contours = measure.find_contours(binary_image, 0.5)

    # Create a blank image for drawing contours
    contour_image = image.copy()
    contour_image[:] = 0

    # Draw contours on the image
    for contour in contours:
        contour = contour.astype(int)
        contour_image[contour[:, 0], contour[:, 1]] = 255

    return contour_image


class ImageRetrievalSystem:
    def __init__(self, image_paths):
        self.index = defaultdict(list)
        self.features = []
        self.image_paths = image_paths

    def extract_shape_features(self, image):
        labeled_image = measure.label(image, connectivity=2)  # Label connected components
        properties = measure.regionprops(labeled_image)

        largest_region = max(properties, key=lambda region: region.area)

        area = largest_region.area

        perimeter = largest_region.perimeter

        eccentricity = largest_region.eccentricity

        major_axis_length = largest_region.major_axis_length

        minor_axis_length = largest_region.minor_axis_length

        compactness = area / perimeter
        circularity = (perimeter ** 2) / area

        aspect_ratio = major_axis_length / minor_axis_length

        shape_features = [area, perimeter, eccentricity, major_axis_length, minor_axis_length,
                          compactness, circularity, aspect_ratio]
        return shape_features

    def extract_texture_features(self, image):
        # Calculate texture features
        glcm = feature.graycomatrix(image.astype(np.uint8), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

        texture_features = [feature.graycoprops(glcm, 'contrast').mean(), feature.graycoprops(glcm, 'energy').mean(),
                            feature.graycoprops(glcm, 'homogeneity').mean(),
                            feature.graycoprops(glcm, 'correlation').mean()]

        return texture_features

    def encode_features(self, features):

        encoded_features = []

        for feature_vector in features:
            # Perform feature encoding (e.g., normalization)
            encoded_feature = feature_vector / np.linalg.norm(feature_vector)

            encoded_features.append(encoded_feature)

        return encoded_features

    def index_features(self, encoded_features):

        self.index = defaultdict(list)

        for image_idx, encoded_feature in enumerate(encoded_features):
            # Add the image index to the inverted index for each feature
            for feature_idx, feature_value in enumerate(encoded_feature):
                self.index[feature_idx].append((feature_value, image_idx))

    def load_images(self):
        self.features = []

        # Get all DICOM files in the directory
        file_extensions = ['.dcm', '.dicom']

        for image_path in self.image_paths:
            # Load DICOM image
            ds = pydicom.dcmread(image_path)
            image = ds.pixel_array

            # Preprocess image if needed
            # ...

            # Extract shape features
            shape_feats = self.extract_shape_features(image)

            # Extract texture features
            texture_feats = self.extract_texture_features(image)

            # Combine shape and texture features
            feature_vector = np.concatenate((shape_feats, texture_feats))
            self.features.append(feature_vector)

        # Encode and index the features
        encoded_features = self.encode_features(self.features)
        self.index_features(encoded_features)

    def query_image(self, image_pixel_data, k=10):
        # Extract shape features
        query_shape_feats = self.extract_shape_features(image_pixel_data)

        # Extract texture features
        query_texture_feats = self.extract_texture_features(image_pixel_data)

        # Combine query shape and texture features
        query_feature_vector = np.concatenate((query_shape_feats, query_texture_feats))

        # Perform the query
        candidates = set()
        for feature_idx, query_value in enumerate(query_feature_vector):
            # Get the candidate images for this feature
            candidates_for_feature = self.index[feature_idx]

            # Find the nearest neighbor for the query value
            nearest_neighbor = min(candidates_for_feature, key=lambda x: abs(x[0] - query_value))

            # Add the image index to the candidate set
            candidates.add(nearest_neighbor[1])

        # Calculate similarity scores
        similarities = [self.cosine_similarity(query_feature_vector, self.features[img_idx]) for img_idx in candidates]

        # Sort the candidate images based on similarity scores
        sorted_candidates = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)

        # Retrieve the top-ranked images
        top_ranked_images = [self.image_paths[image_idx] for image_idx, _ in sorted_candidates[:k]]

        return top_ranked_images

    @staticmethod
    def cosine_similarity(query_vector, database_vector):

        dot_product = np.dot(query_vector, database_vector)

        query_norm = np.linalg.norm(query_vector)

        db_norm = np.linalg.norm(database_vector)

        similarity = dot_product / (query_norm * db_norm)
        return similarity

    @staticmethod
    def evaluate_retrieval(top_ranked_images, ground_truth, k=10):
        retrieved_images = set(top_ranked_images[:k])
        relevant_images = set(ground_truth)
        true_positives = retrieved_images.intersection(relevant_images)
        precision = len(true_positives) / len(retrieved_images) if len(retrieved_images) > 0 else 0.0
        recall = len(true_positives) / len(relevant_images) if len(relevant_images) > 0 else 0.0

        return precision, recall
