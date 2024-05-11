# Intro

This Python script provides functionality to process different types of motion capture file formats, including CSV, BVH, and C3D.

## Folder Structure

TO DO

## Installation

TO DO

## Usage
Run the script with the following command:

```bash
python main.py <file_type>
```

Replace <file_type> with one of the supported file types: "CSV_SKELETON", "CSV_RIGID", "BVH", or "C3D". 


For example:

```bash
python main.py CSV_SKELETON
```


## Supported File Types

- **CSV_SKELETON**: CSV file containing skeleton data.
- **CSV_RIGID**: CSV file containing rigid body data.
- **BVH**: Biovision Hierarchy (BVH) file.
- **C3D**: C3D file format commonly used in motion capture.

## Functionality

The script contains functions to read and process each supported file type:

-   `read_csv_skeleton(filename)`: Reads skeleton data from a CSV file.
-   `read_csv_rigid(filename)`: Reads rigid body data from a CSV file.
-   `read_bvh(filename)`: Reads data from a BVH file.
-   `read_c3d(filename)`: Reads data from a C3D file.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License