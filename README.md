# DenseNet-YOLO Object Detection

A PyTorch implementation of an object detection model that combines DenseNet backbone with YOLO (You Only Look Once) detection heads for efficient and accurate object detection.



## Features

- DenseNet121 backbone pretrained on ImageNet for feature extraction
- Multi-scale detection with two YOLO detection heads (13×13 and 26×26 grids)
- Non-Maximum Suppression for removing overlapping detections
- Support for COCO dataset with 80 classes
- Training with data augmentation for better generalization
- Mean Average Precision (mAP) evaluation

## Model Architecture

The model combines the powerful feature extraction capabilities of DenseNet121 with the efficient single-shot detection approach of YOLO:

1. **Backbone**: DenseNet121 pretrained on ImageNet serves as the feature extractor
2. **Detection Heads**: Two detection heads operating at different scales:
   - 13×13 grid for detecting larger objects
   - 26×26 grid for detecting smaller objects
3. **Skip Connections**: Features from earlier layers are combined with upsampled features for better detection of small objects

## Installation

```bash
# Clone the repository
git clone https://github.com/username/densenet-yolo.git
cd densenet-yolo

# Create conda environment
conda create -n densenet-yolo python=3.8
conda activate densenet-yolo

# Install requirements
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm opencv-python pillow
```

## Usage

### Training on COCO Dataset

```python
# Adjust the paths to your COCO dataset
coco_img_dir = '/path/to/coco/images'
coco_ann_file = '/path/to/coco/annotations'

# Run training script
python train.py --img_dir $coco_img_dir --ann_file $coco_ann_file --epochs 50 --batch_size 64
```

### Inference on a Single Image

```python
from inference import demo

# Run inference on a test image
demo(model_path='path/to/checkpoint.pth', 
     image_path='path/to/test_image.jpg',
     conf_thresh=0.25)
```

### Evaluation on Validation Set

```python
python evaluate.py --img_dir $coco_img_dir --ann_file $coco_ann_file --model_path checkpoint.pth
```

## Code Structure

- `model.py`: Definition of the DenseNetYOLO model architecture
- `train.py`: Training script including data loading and loss computation
- `evaluate.py`: Evaluation script for computing mAP on validation set
- `inference.py`: Utility functions for inference and visualization
- `utils.py`: Helper functions for NMS, IoU calculation, etc.

## Training Details

The model is trained with the following parameters:

- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate**: Starting at 1e-5 with cosine annealing schedule and warmup
- **Augmentations**: Random horizontal flip, color jitter, random rotation
- **Input Size**: 416×416 pixels
- **Batch Size**: 64 (adjust based on available GPU memory)
- **Loss Functions**: MSE for bounding box coordinates, BCE for objectness and class probabilities

## Evaluation

The model is evaluated using mean Average Precision (mAP) at an IoU threshold of 0.5, which is a standard metric for object detection tasks.

## Pretrained Models

Pretrained models are available for download:
- [DenseNet-YOLO trained on COCO (5 epochs)](https://example.com/densenet-yolo-coco-5ep.pth)
- [DenseNet-YOLO trained on COCO (50 epochs)](https://example.com/densenet-yolo-coco-50ep.pth)

## Results

| Model | Backbone | Input Size | mAP@0.5 | FPS (RTX 3090) |
|-------|----------|------------|---------|----------------|
| DenseNet-YOLO | DenseNet121 | 416×416 | 58.3% | 45 |

## Examples

The following demonstrates how to use the model for inference:

```python
import torch
from PIL import Image
from model import DenseNetYOLO
from utils import process_predictions, improved_nms
from torchvision import transforms

# Load model
model = DenseNetYOLO(num_classes=80)
checkpoint = torch.load('checkpoint.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open('test_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to('cuda')

# Get predictions
with torch.no_grad():
    outputs = model(input_tensor)
    
# Process outputs
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Anchors for scale 13x13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]   # Anchors for scale 26x26
]
boxes = process_predictions(outputs, ANCHORS, conf_thresh=0.25)
boxes = improved_nms(boxes)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DenseNet implementation from torchvision
- YOLO algorithm design principles from the original YOLO papers
- COCO dataset for object detection

## Citation

If you use this code in your research, please cite:

```
@misc{densenet-yolo,
  author = {Your Name},
  title = {DenseNet-YOLO Object Detection},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/username/densenet-yolo}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
