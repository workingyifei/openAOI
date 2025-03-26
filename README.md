# OpenAOI - PCB/A Defect Detection System

An open-source Automated Optical Inspection (AOI) system for detecting and measuring PCB/A defects using computer vision and deep learning. The system currently focuses on solder paste and bridging defects but is designed to be expandable to cover the full range of IPC-610 defect classes.

## Current Scope

The system currently handles two specific defect types from our dataset:
- **Solder Paste Slope Defects** (`Bad_slope`, formerly 坡度/pōdù)
- **Solder Bridging Defects** (`Bad_bridge`, formerly 桥脚/qiáojiǎo)

The dataset was found online at [Kaggle](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) and can be accessed for further details and downloads.

These defects were chosen based on the available defects in the dataset, which contains high-quality images captured using industrial inspection systems. The images show both solder paste and mounted components, suggesting they were captured using:
- RGB cameras with high resolution
- Angled lighting to enhance solder paste texture visibility
- Possibly IR components for better solder joint visualization

### Future Expansion

While the current implementation focuses on two defect types, the system is designed to be extended to cover the full range of IPC-610 defect classes, including:

1. Additional Solder Joint Defects
   - Insufficient Solder: This defect occurs when there is not enough solder applied to the joint, leading to weak connections. Possible root causes include inadequate solder paste application or improper reflow temperatures.
   - Excess Solder: This defect is characterized by an overabundance of solder, which can cause bridging between joints. Root causes may include excessive solder paste application or incorrect stencil design.
   - Voiding: Voids are air pockets trapped under the solder joint, which can compromise electrical connectivity. They may result from improper solder paste printing, insufficient reflow time, or contamination.

2. Component Placement Defects
   - Tombstoning: This defect happens when one end of a component lifts off the PCB during soldering, often due to uneven heating or improper solder volume. It can lead to unreliable connections.
   - Component Shift: This occurs when components are misaligned during placement, which can result from mechanical issues in the pick-and-place machine or incorrect programming.
   - Component Rotation: This defect involves components being rotated incorrectly on the PCB, potentially due to misalignment during placement or issues with the component's orientation in the feeder.

3. Surface Defects
   - Contamination: Contaminants on the PCB surface can interfere with solder adhesion, leading to weak joints. Common causes include dust, oils, or residues from previous manufacturing processes.
   - Solder Balls: These are small spheres of solder that can form during the soldering process, often due to excessive solder or improper heating. They can create shorts if they land on adjacent pads.

4. PCB and Pad Defects
   - Pad Lift: This defect occurs when the pad lifts off the PCB substrate, often due to thermal stress or poor adhesion. It can lead to open circuits and unreliable connections.
   - Copper Exposure: This defect involves exposed copper areas on the PCB, which can lead to corrosion or short circuits. It may result from inadequate solder mask application or damage during handling.

5. Lead Defects
   - Lead Bend: This defect occurs when component leads are bent, which can complicate placement and soldering. It may be caused by mishandling or improper storage.
   - Lead Protrusion: This defect involves leads extending beyond the PCB surface, which can lead to soldering issues or mechanical failures. It may result from incorrect lead length or placement errors.

## Features

- **Defect Detection**: YOLO-based model for PCB/A defect detection
- **Measurement Tools**: Precise measurements of defect characteristics
- **Visualization Tools**: Rich visualization options for annotations and measurements
- **Evaluation Tools**: Comprehensive model evaluation and metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/workingyifei/openAOI.git
cd openAOI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The system uses a dataset of pre-captured PCB/A inspection images. These images were captured using industrial AOI systems with the following characteristics:

- High-resolution RGB images
- Controlled lighting conditions
- Multiple viewing angles
- Possible IR imaging components

### Dataset Structure

```
dataset/
├── train_data/
│   ├── Annotations/          # XML annotation files
│   ├── JPEGImages/          # Source images
│   └── index.txt            # Image-annotation pairs
├── test_data/               # Test dataset
└── train_data_augmentation/ # Augmented training data
```

## Usage

### 1. Visualizing Annotations

```bash
python src/tools/visualize_annotations.py \
    --dataset path/to/dataset \
    --output visualization_results
```

This will:
- Load inspection images and their annotations
- Draw white bounding boxes around detected defects
- Display color-coded defect labels (yellow for Bad_slope, magenta for Bad_bridge)
- Show comprehensive measurements with smart label placement
- Save visualized results with clear annotations

The visualization includes:
- For `Bad_slope`:
  - Aspect ratio (industry standard: 1.5-1.8:1)
  - Coverage percentage
  - Approximate slope angle (with limitations noted)
  - Paste uniformity
- For `Bad_bridge`:
  - Width in pixels
  - Aspect ratio
  - Bridge count
  - Severity percentage

Features:
- Smart label placement to avoid overlaps
- Leader lines connecting labels to defects when needed
- Black backgrounds behind text for readability
- Color-coded text for defect types
- Multi-line measurement display

### 2. Measuring Defects

```bash
python src/tools/measure_defects.py \
    --image path/to/image.jpg \
    --bbox 0.5 0.5 0.2 0.2 \
    --type Bad_slope \
    --output measured_defect.jpg
```

Measurements include:
- For `Bad_slope`:
  - Geometric properties:
    - Aspect ratio (compared to industry standard 1.5-1.8:1)
    - Coverage percentage using Otsu's thresholding
    - Area in pixels
  - Quality indicators:
    - Approximate slope angle (from top-down view)*
    - Paste uniformity based on intensity variation
    - Mean and standard deviation of intensity
  - *Note: Accurate slope measurement requires side-view or 3D data

- For `Bad_bridge`:
  - Geometric measurements:
    - Width and length in pixels
    - Aspect ratio
    - Area calculation
  - Quality assessment:
    - Bridge count using contour analysis
    - Severity percentage based on intensity profile
    - Bridge completeness evaluation

### 3. Evaluating Model Performance

```bash
python src/tools/evaluate_model.py \
    --model path/to/model.pt \
    --test-data path/to/test/data \
    --output evaluation_results \
    --conf-thres 0.25
```

Generates:
- Precision, recall, and F1-score metrics
- Confusion matrices
- Confidence score distributions
- Inference time analysis
- Annotated test images

## Model Configuration

The model configuration (`config/model_config.yaml`) includes:
- Model: YOLOv8
- Input size: 640x640
- Training parameters
- Data augmentation settings
- Class definitions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Adding New Defect Types

The system is designed to be extended with additional IPC-610 defect classes:

1. Add new defect class to model configuration
2. Update measurement tools for new defect characteristics
3. Add specific visualization methods
4. Extend evaluation metrics as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details. 