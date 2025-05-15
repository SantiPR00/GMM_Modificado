
# GMM-based Semantic Segmentation (Modified Version)

This repository is based on an existing public implementation of road segmentation using various classifiers, including Gaussian Mixture Models (GMM). 
Original source: [https://github.com/bhargavab17/Road-Segmentation-using-different-classifiers](https://github.com/bhargavab17/Road-Segmentation-using-different-classifiers)

This modified version was developed as part of a Bachelor's Thesis focused on the evaluation, adaptation, and improvement of semantic segmentation algorithms in robotics environments. The work involved resolving compatibility issues and restructuring the code for better integration with ROS and visualization tools.

## Overview of Modifications

- Code refactoring for clarity and modularity
- Improved handling of image data for segmentation input and output
- Adjusted segmentation parameters for comparative evaluation
- Added scripts and configuration for simplified execution

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/SantiPR00/GMM_Modificado.git
cd GMM_Modificado
```

2. Ensure Python 3 and OpenCV are installed. This implementation does not require GPU acceleration or deep learning frameworks.

3. Run the segmentation script on a sample image:

```bash
python3 segment_image_gmm.py
```

4. The output will be saved as a color-mapped image generated on the same directory of the repository in a .png called 'segmentation.png'

## Folder Structure

```
GMM_Modificado/
├── src/
│   └── Road-Segmentation-using-different-classifiers/
│       ├── segment_image_gmm.py
│       ├── utils/
│       └── images/
├── outputs/
├── scripts/
```

## Notes

This project demonstrates a lightweight and interpretable approach to semantic segmentation using statistical models. It serves as a baseline for comparison with more complex methods based on deep learning.

## Author

This version was prepared and modified by [SantiPR00](https://github.com/SantiPR00) as part of an academic research project in semantic scene understanding.
