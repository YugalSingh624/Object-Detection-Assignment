import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

NUM_CLASSES = len(COCO_CLASSES)
IMG_SIZE = 416  # YOLO standard size
GRID_SIZES = [13, 26]  # Different grid sizes for multi-scale detection

# Anchors (these should match what you used during training)
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Anchors for scale 13x13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]   # Anchors for scale 26x26
]

# Model definition - this needs to match your training code exactly
class DenseNetYOLO(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DenseNetYOLO, self).__init__()
        # Load DenseNet121 backbone
        from torchvision.models import densenet121, DenseNet121_Weights
        densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Initial convolution and pooling
        self.conv0 = densenet.features.conv0
        self.norm0 = densenet.features.norm0
        self.relu0 = densenet.features.relu0
        self.pool0 = densenet.features.pool0
        
        # Dense blocks and transition layers
        self.denseblock1 = densenet.features.denseblock1
        self.transition1 = densenet.features.transition1
        self.denseblock2 = densenet.features.denseblock2
        self.transition2 = densenet.features.transition2
        self.denseblock3 = densenet.features.denseblock3
        self.transition3 = densenet.features.transition3
        self.denseblock4 = densenet.features.denseblock4
        self.norm5 = densenet.features.norm5
        
        # Number of output channels at each feature level
        self.channels = {
            'transition1': 128,  # After first transition
            'transition2': 256,  # After second transition
            'final': 1024        # After final dense block
        }
        
        # Detection head for scale 1 (13x13)
        self.detect1_conv = nn.Sequential(
            nn.Conv2d(self.channels['final'], 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.detect1_output = nn.Conv2d(512, len(ANCHORS[0]) * (5 + num_classes), kernel_size=1)
        
        # Upsample for scale 2
        self.upsample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # Detection head for scale 2 (26x26)
        self.detect2_conv = nn.Sequential(
            nn.Conv2d(256 + self.channels['transition2'], 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.detect2_output = nn.Conv2d(256, len(ANCHORS[1]) * (5 + num_classes), kernel_size=1)
    
    def forward(self, x):
        # Store intermediate outputs for skip connections
        feature_maps = {}
        
        # Initial layers
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        # Dense block 1 and transition 1
        x = self.denseblock1(x)
        x = self.transition1(x)
        feature_maps['transition1'] = x  # Save for possible future use
        
        # Dense block 2 and transition 2
        x = self.denseblock2(x)
        x = self.transition2(x)
        feature_maps['transition2'] = x  # Save for skip connection
        
        # Dense block 3 and transition 3
        x = self.denseblock3(x)
        x = self.transition3(x)
        
        # Dense block 4 and final norm
        x = self.denseblock4(x)
        x = self.norm5(x)
        feature_maps['final'] = x
        
        # Detection at scale 1 (13x13)
        detect1_features = self.detect1_conv(x)
        output1 = self.detect1_output(detect1_features)
        
        # Upsample for scale 2 (26x26)
        upsampled = self.upsample(detect1_features)
        
        # Ensure the feature maps have compatible dimensions for concatenation
        transition2 = feature_maps['transition2']
        if upsampled.shape[2:] != transition2.shape[2:]:
            # Resize if dimensions don't match
            transition2 = F.interpolate(transition2, size=upsampled.shape[2:], mode='nearest')
        
        # Concatenate upsampled features with transition2 features
        concat_features = torch.cat([upsampled, transition2], dim=1)
        
        # Detection at scale 2
        detect2_features = self.detect2_conv(concat_features)
        output2 = self.detect2_output(detect2_features)
        
        return [output1, output2]

# Function to process model outputs to bounding boxes
def process_predictions(outputs, anchors, img_size=IMG_SIZE, conf_thresh=0.25):
    """
    Process YOLO outputs to get bounding boxes and class predictions.
    Args:
        outputs: List of tensors, one for each scale
        anchors: List of anchors for each scale
        img_size: Input image size
        conf_thresh: Confidence threshold
    Returns:
        Tensor of detected boxes (batch_size, num_boxes, 6) where each box contains [x1, y1, x2, y2, conf, class_idx]
    """
    all_boxes = []
    for i, output in enumerate(outputs):
        # Get grid size from output shape
        batch_size, num_anchors_and_attrs, grid_size, _ = output.shape
        
        # Number of attributes per anchor: 5 (x, y, w, h, obj) + num_classes
        num_attrs = (output.shape[1] // len(anchors[i]))
        num_classes = num_attrs - 5
        
        # Reshape output
        prediction = output.view(batch_size, len(anchors[i]), num_attrs, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Sigmoid object confidence and class scores
        prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])
        
        # Process each image in the batch
        for b in range(batch_size):
            # Only select boxes above confidence threshold
            obj_mask = prediction[b, ..., 4] > conf_thresh
            if not obj_mask.any():
                continue
                
            # Get predicted boxes
            pred_boxes = prediction[b, obj_mask]
            
            # Get box attributes
            box_attr = torch.zeros_like(pred_boxes)
            
            # Get grid positions
            grid_indices = obj_mask.nonzero()
            anchor_idx = grid_indices[:, 0]
            grid_y = grid_indices[:, 1]
            grid_x = grid_indices[:, 2]
            
            # Apply anchor box offsets and scale
            box_attr[:, 0] = (torch.sigmoid(pred_boxes[:, 0]) + grid_x) / grid_size
            box_attr[:, 1] = (torch.sigmoid(pred_boxes[:, 1]) + grid_y) / grid_size
            
            # Width and height
            anchor_w = torch.tensor([a[0] for a in anchors[i]], device=device)[anchor_idx]
            anchor_h = torch.tensor([a[1] for a in anchors[i]], device=device)[anchor_idx]
            box_attr[:, 2] = torch.exp(pred_boxes[:, 2]) * anchor_w
            box_attr[:, 3] = torch.exp(pred_boxes[:, 3]) * anchor_h
            
            # Convert to corner coordinates (x1, y1, x2, y2)
            box_attr[:, 0:2] -= box_attr[:, 2:4] / 2
            box_attr[:, 2:4] += box_attr[:, 0:2]
            
            # Scale to image size
            box_attr[:, 0:4] *= img_size
            
            # Add confidence scores
            box_attr[:, 4] = pred_boxes[:, 4]
            
            # Add class with highest probability
            box_attr[:, 5] = pred_boxes[:, 5:].argmax(1)
            
            all_boxes.append(box_attr)
    
    # Combine all boxes from different scales
    if len(all_boxes) > 0:
        return torch.cat(all_boxes, dim=0)
    else:
        return torch.zeros((0, 6), device=device)

# Improved NMS implementation
def improved_nms(boxes, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on the bounding boxes.
    Args:
        boxes: Tensor of shape (N, 6) where each row is (x1, y1, x2, y2, confidence, class_idx)
        iou_threshold: IoU threshold for boxes to be considered overlapping
    Returns:
        Tensor of shape (M, 6) where M <= N, containing filtered boxes
    """
    # If no boxes, return empty tensor
    if boxes.numel() == 0:
        return torch.zeros((0, 6), device=boxes.device)
    
    # Extract coordinates, scores, and class indices
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    scores = boxes[:, 4]
    class_idxs = boxes[:, 5]
    
    # Calculate areas for all boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort boxes by confidence scores (high to low)
    _, order = scores.sort(descending=True)
    
    keep_boxes = []
    while order.numel() > 0:
        # Pick the box with highest confidence score
        i = order[0].item()
        
        # Add it to the list of kept boxes
        keep_boxes.append(boxes[i])
        
        # If this was the last box, break
        if order.numel() == 1:
            break
            
        # Calculate IoU of the picked box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        # Calculate intersection area
        width = torch.clamp(xx2 - xx1, min=0)
        height = torch.clamp(yy2 - yy1, min=0)
        intersection = width * height
        
        # Calculate IoU
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Apply stricter filtering: Keep boxes with IoU less than threshold,
        # even if they're from different classes to reduce crowding
        mask = iou <= iou_threshold
        
        # Update order indices
        order = order[1:][mask]
    
    if keep_boxes:
        return torch.stack(keep_boxes)
    else:
        return torch.zeros((0, 6), device=boxes.device)

# Setup transformation for input images
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_frame(frame):
    # Convert BGR (OpenCV format) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms
    transform = get_transform()
    input_tensor = transform(pil_image)
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    return input_tensor

def run_detection(model_path, conf_threshold=0.6, iou_threshold=0.3):
    # Load model
    model = DenseNetYOLO(num_classes=NUM_CLASSES)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define random colors for each class (for visualization)
    np.random.seed(42)  # For reproducible colors
    colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
    
    print("Starting real-time detection. Press 'q' to exit.")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Add a counter to limit the maximum number of boxes displayed
    max_boxes_to_display = 15
    
    with torch.no_grad():
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:  # Update FPS every 10 frames
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame)
            
            # Start time for inference
            start_time = time.time()
            
            # Forward pass
            outputs = model(input_tensor)
            
            # Process predictions - use higher threshold for initial detection
            detections = process_predictions(outputs, ANCHORS, conf_thresh=conf_threshold)
            detections = improved_nms(detections, iou_threshold=iou_threshold)
            
            # End time for inference
            inference_time = time.time() - start_time
            
            # Draw bounding boxes on frame
            if detections.size(0) > 0:
                # Scale detections to original frame size
                scale_x = frame_width / IMG_SIZE
                scale_y = frame_height / IMG_SIZE
                
                # Sort by confidence (highest first) and limit display
                conf_scores = detections[:, 4]
                _, conf_idx = torch.sort(conf_scores, descending=True)
                
                # Limit to top N most confident detections
                conf_idx = conf_idx[:max_boxes_to_display]
                
                for idx in conf_idx:
                    # Make sure we only take the first 6 elements
                    detection_values = detections[idx].cpu().numpy()
                    x1, y1, x2, y2, conf, class_id = detection_values[:6]
                    
                    # Scale coordinates
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    class_id = int(class_id)
                    
                    # Get class name and color
                    class_name = COCO_CLASSES[class_id]
                    color = colors[class_id].tolist()
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label text - Include confidence in label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Draw label background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show inference time and FPS
            cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Real-time Object Detection", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "yolo_densenet_checkpoint_epoch2.pth"
    run_detection(model_path, conf_threshold=0.8, iou_threshold=0.5)