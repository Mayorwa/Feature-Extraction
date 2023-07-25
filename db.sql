-- Table: Texture Features
CREATE TABLE TextureFeatures (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_id INT NOT NULL,
    contrast FLOAT,
    homogeneity FLOAT,
    energy FLOAT,
    correlation FLOAT,
    FOREIGN KEY (image_id) REFERENCES DICOMTags(id)
);

-- Table: Shape Features
CREATE TABLE ShapeFeatures (
    id INT PRIMARY KEY AUTO_INCREMENT,
    image_id INT NOT NULL,
    area FLOAT,
    perimeter FLOAT,
    compactness FLOAT,
    circularity FLOAT,
    eccentricity FLOAT,
    FOREIGN KEY (image_id) REFERENCES DICOMTags(id)
);

-- Table: DICOM Tags
CREATE TABLE DICOMTags (
    id INT PRIMARY KEY AUTO_INCREMENT,
    modality VARCHAR(50),
    study_description VARCHAR(255),
    study_date DATE,
    image_type VARCHAR(255),
    body_part_examined VARCHAR(100),
    -- Additional DICOM tags as needed
);
