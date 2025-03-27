# Experimental Features

This directory contains experimental features that are not currently part of the main workflow but may be useful for future development.

## Camera Acquisition (`acquisition/`)

Contains code for real-time camera capture using Basler cameras with advanced features:
- Multi-angle capture
- Polarization control
- Real-time image processing

### Usage
This code is currently not integrated into the main workflow. To use it:
1. Install the required dependencies (pypylon)
2. Configure your camera settings in the config file
3. Use the `InspectionCamera` class to capture images

### Future Integration
To integrate this with the main workflow:
1. Update the camera settings in `config/model_config.yaml`
2. Modify `src/main.py` to use the camera capture code
3. Update the data pipeline to handle real-time images 