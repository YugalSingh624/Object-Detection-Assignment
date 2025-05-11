import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
import torchvision
from PIL import Image


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

NUM_CLASSES = len(COCO_CLASSES)

# Constants
IMG_SIZE = 416  # YOLO standard size
GRID_SIZES = [13, 26]  # Different grid sizes for multi-scale detection

# Anchors (these should be optimized for your dataset using k-means clustering)
# Format: width, height (normalized by image size)
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # Anchors for scale 13x13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]   # Anchors for scale 26x26
]


# DenseNet Backbone with YOLO Detection Heads
# DenseNet Backbone with YOLO Detection Heads
class DenseNetYOLO(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DenseNetYOLO, self).__init__()
        
        # Load DenseNet121 backbone
        # Note: Don't use the backbone directly to avoid feature extraction issues
        densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # DenseNet121 feature extractor layers - manually defined
        # This avoids the complex feature extraction logic and ensures we have
        # proper control over the layer outputs and feature maps
        
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
        
        # Freeze early layers
        freeze_layers = [
            self.conv0, self.norm0, self.relu0, self.pool0,
            self.denseblock1, self.transition1,
            self.denseblock2, self.transition2
        ]
        
        for layer in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
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


# Custom dataset class for COCO
class COCODataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        """
        A simple COCO dataset implementation.
        
        Args:
            img_dir: Directory containing images
            annotations_file: Path to COCO annotations JSON file
            transform: Image transformations
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # In a real implementation, you would use pycocotools to load annotations
        # For simplicity, we're assuming a simple structure here
        # You should replace this with proper COCO annotation loading
        import json
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image IDs
        self.img_ids = [img['id'] for img in self.annotations['images']]
        
        # Create image ID to annotations mapping
        self.id_to_annotations = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_annotations:
                self.id_to_annotations[img_id] = []
            self.id_to_annotations[img_id].append(ann)
            
        # Create image ID to filename mapping
        self.id_to_filename = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        # Category ID to class index mapping
        self.cat_id_to_class = {cat['id']: i for i, cat in enumerate(self.annotations['categories'])}
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_filename = self.id_to_filename[img_id]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Get annotations for this image
        anns = self.id_to_annotations.get(img_id, [])
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert bounding boxes to YOLO format:
        # [class_idx, x_center, y_center, width, height] - normalized coordinates
        targets = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
            
            # Convert to YOLO format
            # Normalize coordinates
            x_center = (bbox[0] + bbox[2] / 2) / orig_width
            y_center = (bbox[1] + bbox[3] / 2) / orig_height
            width = bbox[2] / orig_width
            height = bbox[3] / orig_height
            
            # Get class index
            class_idx = self.cat_id_to_class[ann['category_id']]
            
            # Only add if box is valid
            if width > 0 and height > 0:
                targets.append([class_idx, x_center, y_center, width, height])
        
        # Convert to tensor
        targets = torch.tensor(targets, dtype=torch.float32)
        
        return image, targets


# Function to convert model outputs to bounding boxes
def process_predictions(outputs, anchors, img_size=IMG_SIZE, conf_thresh=0.25):
    """
    Process YOLO outputs to get bounding boxes and class predictions.
    
    Args:
        outputs: List of tensors, one for each scale
        anchors: List of anchors for each scale
        img_size: Input image size
        conf_thresh: Confidence threshold
    
    Returns:
        List of detected boxes (batch_size, num_boxes, 6) where each box contains
        [x1, y1, x2, y2, conf, class_idx]
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
            anchor_idx, grid_y, grid_x = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]
            
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


# Non-Maximum Suppression to remove overlapping boxes
def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: Tensor of shape (N, 6) with each box as [x1, y1, x2, y2, conf, class_idx]
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Tensor of shape (M, 6) with M <= N
    """
    if boxes.size(0) == 0:
        return boxes
    
    # Sort by confidence
    _, order = boxes[:, 4].sort(descending=True)
    boxes = boxes[order]
    
    keep = []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    while boxes.size(0) > 0:
        i = 0  # Keep the box with highest confidence
        keep.append(order[i])
        
        if boxes.size(0) == 1:
            break
            
        # Get IoU of the box with highest confidence with all other boxes
        xx1 = torch.max(boxes[0, 0], boxes[1:, 0])
        yy1 = torch.max(boxes[0, 1], boxes[1:, 1])
        xx2 = torch.min(boxes[0, 2], boxes[1:, 2])
        yy2 = torch.min(boxes[0, 3], boxes[1:, 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        
        inter = w * h
        iou = inter / (area[0] + area[1:] - inter)
        
        # Keep boxes with IoU below threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        boxes = boxes[inds + 1]
        area = area[inds + 1]
    
    return torch.stack([boxes[i] for i in keep])


# YOLO loss function
class YOLOLoss(nn.Module):
    def __init__(self, anchors, grid_sizes, num_classes=NUM_CLASSES):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.lambda_coord = 5.0  # Weight for box coordinate loss
        self.lambda_noobj = 0.5  # Weight for no-object confidence loss
        self.mse_loss = nn.MSELoss(reduction='mean')  # Changed to mean
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # Changed to BCEWithLogitsLoss for stability
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss with improved stability.
        
        Args:
            predictions: List of tensors from model output
            targets: List of tensors with ground truth boxes for each image
                     Each box is [class_idx, x_center, y_center, width, height]
        
        Returns:
            Total loss
        """
        batch_size = len(targets)
        if batch_size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        total_loss = 0
        coord_loss, obj_loss, noobj_loss, cls_loss = 0, 0, 0, 0
        num_scale_losses = 0
        
        # Process each scale separately
        for scale_idx, (pred, grid_size) in enumerate(zip(predictions, self.grid_sizes)):
            batch_size = pred.size(0)
            num_anchors = len(self.anchors[scale_idx])
            
            # Reshape predictions
            pred = pred.view(batch_size, num_anchors, self.num_classes + 5, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Create target tensors
            obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, device=device)
            tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
            tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes, device=device)
            
            target_count = 0  # Count of valid targets assigned at this scale
            
            # Process each image in the batch
            for b in range(batch_size):
                if b >= len(targets) or targets[b].size(0) == 0:
                    continue
                    
                this_img_targets = targets[b]
                
                # Process each target box
                for target_box in this_img_targets:
                    # Skip invalid classes
                    class_idx = target_box[0].long().item()
                    if class_idx < 0 or class_idx >= self.num_classes:
                        continue
                        
                    # Convert target box to grid scale
                    gx = target_box[1] * grid_size
                    gy = target_box[2] * grid_size
                    gw = target_box[3] * grid_size
                    gh = target_box[4] * grid_size
                    
                    # Grid cell indices (center of box)
                    gi = int(gx)
                    gj = int(gy)
                    
                    # Check if grid indices are within bounds
                    if gi >= grid_size or gj >= grid_size or gi < 0 or gj < 0:
                        continue
                    
                    # Calculate best anchor based on IoU
                    target_box_tensor = torch.tensor([0, 0, target_box[3], target_box[4]], device=device)
                    anchor_ious = []
                    
                    for anchor_idx in range(num_anchors):
                        anchor_box = torch.tensor(
                            [0, 0, self.anchors[scale_idx][anchor_idx][0], self.anchors[scale_idx][anchor_idx][1]],
                            device=device
                        )
                        anchor_ious.append(bbox_iou(anchor_box, target_box_tensor, x1y1x2y2=False))
                    
                    anchor_ious = torch.tensor(anchor_ious, device=device)
                    best_anchor = torch.argmax(anchor_ious).item()
                    
                    # Only assign target if IoU is good enough
                    if anchor_ious[best_anchor] > 0.2:
                        # Mark grid cell as having an object
                        obj_mask[b, best_anchor, gj, gi] = 1
                        noobj_mask[b, best_anchor, gj, gi] = 0
                        
                        # Box coordinates - the fractions for tx, ty
                        tx[b, best_anchor, gj, gi] = gx - gi
                        ty[b, best_anchor, gj, gi] = gy - gj
                        
                        # Width and height - log scale (same as YOLOv3)
                        tw[b, best_anchor, gj, gi] = torch.log(
                            gw / self.anchors[scale_idx][best_anchor][0] + 1e-16
                        )
                        th[b, best_anchor, gj, gi] = torch.log(
                            gh / self.anchors[scale_idx][best_anchor][1] + 1e-16
                        )
                        
                        # Class probability
                        tcls[b, best_anchor, gj, gi, class_idx] = 1
                        
                        target_count += 1
            
            # Skip loss calculation if no targets assigned at this scale
            if target_count == 0:
                continue
                
            num_scale_losses += 1
            
            # Mask for cells with assigned targets
            obj_count = obj_mask.sum()
            noobj_count = noobj_mask.sum()
            
            # Box coordinate loss
            box_pred_xy = torch.sigmoid(pred[..., 0:2][obj_mask == 1])
            box_target_xy = torch.stack([
                tx[obj_mask == 1],
                ty[obj_mask == 1]
            ], dim=1)
            
            if box_pred_xy.numel() > 0:  # Only calculate if we have predictions
                coord_loss += self.mse_loss(box_pred_xy, box_target_xy)
                
                # Width and height loss
                box_pred_wh = pred[..., 2:4][obj_mask == 1]
                box_target_wh = torch.stack([
                    tw[obj_mask == 1],
                    th[obj_mask == 1]
                ], dim=1)
                
                coord_loss += self.mse_loss(box_pred_wh, box_target_wh)
            
            # Objectness loss (using BCE with logits for stability)
            obj_loss += self.bce_loss(
                pred[..., 4][obj_mask == 1],
                torch.ones_like(pred[..., 4][obj_mask == 1])
            )
            
            # No-object loss (using BCE with logits for stability)
            # Limit the number of no-object examples to balance the loss
            if noobj_count > 0:
                # Balance: use at most 3x the number of objects
                max_noobj = min(int(obj_count * 3), int(noobj_count))
                if max_noobj > 0:
                    noobj_mask_balanced = (noobj_mask.flatten() > 0).nonzero().squeeze()
                    # Randomly sample indices if we have too many
                    if noobj_mask_balanced.numel() > max_noobj:
                        idx = torch.randperm(noobj_mask_balanced.numel())[:max_noobj]
                        noobj_mask_balanced = noobj_mask_balanced[idx]
                    
                    flat_pred = pred[..., 4].flatten()
                    noobj_pred = flat_pred[noobj_mask_balanced]
                    
                    noobj_loss += self.bce_loss(
                        noobj_pred, 
                        torch.zeros_like(noobj_pred)
                    )
            
            # Class loss
            if obj_count > 0:
                cls_pred = pred[..., 5:][obj_mask == 1]
                cls_target = tcls[obj_mask == 1]
                
                if cls_pred.numel() > 0:
                    cls_loss += self.bce_loss(cls_pred, cls_target)
        
        # Combine losses with weights
        if num_scale_losses > 0:
            total_loss = (
                self.lambda_coord * coord_loss +
                obj_loss +
                self.lambda_noobj * noobj_loss +
                cls_loss
            ) / num_scale_losses
        else:
            # Return a small loss to prevent NaN gradients if no targets
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        return total_loss


# Improved NMS implementation
def improved_nms(boxes, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on the bounding boxes.
    
    Args:
        boxes: Tensor of shape (N, 6) where each row is
               (x1, y1, x2, y2, confidence, class_idx)
        iou_threshold: IoU threshold for boxes to be considered overlapping
        
    Returns:
        Tensor of shape (M, 6) where M <= N, containing filtered boxes
    """
    # If no boxes, return empty tensor
    if boxes.numel() == 0:
        return torch.zeros((0, 6), device=boxes.device)
    
    # Make sure boxes is 2D tensor with shape (N, 6)
    if len(boxes.shape) == 1:
        # If we have a single box, reshape to (1, 6)
        boxes = boxes.unsqueeze(0)
    
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
        box_to_keep = boxes[i]
        
        # Ensure box_to_keep is a 1D tensor
        if len(box_to_keep.shape) > 1:
            box_to_keep = box_to_keep.squeeze(0)
        
        keep_boxes.append(box_to_keep)
        
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
        
        # Keep boxes with IoU less than threshold and same class
        same_class = class_idxs[order[1:]] == class_idxs[i]
        below_thresh = iou <= iou_threshold
        mask = below_thresh | ~same_class
        
        # Update order indices
        order = order[1:][mask]
    
    # Stack kept boxes into a single tensor
    # First ensure all tensors have the same shape (1D with 6 elements)
    for i in range(len(keep_boxes)):
        if len(keep_boxes[i].shape) > 1:
            keep_boxes[i] = keep_boxes[i].squeeze(0)
    
    if keep_boxes:
        return torch.stack(keep_boxes)
    else:
        return torch.zeros((0, 6), device=boxes.device)


# Efficient IoU calculation for batched boxes
def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4)
        box2: Tensor of shape (M, 4)
    
    Returns:
        IoU: Tensor of shape (N, M)
    """
    # Get box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Get the coordinates of intersecting rectangles
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    # Calculate intersection area
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-16)
    
    return iou


# Calculate IoU between two bounding boxes
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Calculate IoU between two boxes.
    
    Args:
        box1: Tensor of shape (4,)
        box2: Tensor of shape (4,)
        x1y1x2y2: If True, box format is [x1, y1, x2, y2], otherwise [x, y, w, h]
    
    Returns:
        IoU score
    """
    if not x1y1x2y2:
        # Transform from [x, y, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    else:
        # Get box coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    # Get intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-16)
    
    return iou


# Calculate mAP for evaluation
# Calculate mAP for evaluation
# Calculate mAP for evaluation
def calculate_mAP(pred_boxes, true_boxes, num_classes=NUM_CLASSES, iou_threshold=0.5):
    """
    Calculate mean Average Precision for object detection.
    
    Args:
        pred_boxes: List of tensors, each with shape [num_boxes, 6] (x1,y1,x2,y2,conf,class)
        true_boxes: List of tensors, each with shape [num_boxes, 5] (class,x,y,w,h)
        num_classes: Number of classes
        iou_threshold: IoU threshold for a detection to be considered correct
        
    Returns:
        mAP score
    """
    # Initialize counters and storage
    average_precisions = []
    
    # Track if we have any valid predictions
    has_valid_predictions = False
    
    # Debug counters
    total_predictions = 0
    valid_predictions = 0
    
    # Process each class
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # Get all detections and ground truths for this class
        for img_idx in range(len(pred_boxes)):
            # Get predictions for this class (if any exist for this image)
            if img_idx < len(pred_boxes) and pred_boxes[img_idx].size(0) > 0:
                # Extract boxes for this class
                pred = pred_boxes[img_idx]
                mask = (pred[:, 5] == c)
                total_predictions += mask.sum().item()
                
                if mask.sum() > 0:
                    valid_predictions += mask.sum().item()
                    has_valid_predictions = True
                    
                    # Store detections
                    for box, conf in zip(pred[mask, :4], pred[mask, 4]):
                        detections.append({
                            'image_id': img_idx,
                            'confidence': conf.item(),
                            'bbox': box.cpu().numpy()
                        })
            
            # Get ground truths for this class (if any exist for this image)
            if img_idx < len(true_boxes) and true_boxes[img_idx].size(0) > 0:
                gt = true_boxes[img_idx]
                
                # Check if gt is a 1D tensor (single box case)
                if gt.dim() == 1:
                    # Single box case - check if this box is for class c
                    if gt[0] == c:
                        # Convert from normalized [x_center, y_center, width, height] to absolute [x1, y1, x2, y2]
                        x, y, w, h = gt[1:].cpu().numpy()
                        x1 = (x - w / 2) * IMG_SIZE
                        y1 = (y - h / 2) * IMG_SIZE
                        x2 = (x + w / 2) * IMG_SIZE
                        y2 = (y + h / 2) * IMG_SIZE
                        ground_truths.append({
                            'image_id': img_idx,
                            'bbox': np.array([x1, y1, x2, y2])
                        })
                else:
                    # Multiple boxes case
                    class_mask = (gt[:, 0] == c)
                    if class_mask.sum() > 0:
                        for box in gt[class_mask, 1:]:
                            # Convert from normalized [x_center, y_center, width, height] to absolute [x1, y1, x2, y2]
                            x, y, w, h = box.cpu().numpy()
                            x1 = (x - w / 2) * IMG_SIZE
                            y1 = (y - h / 2) * IMG_SIZE
                            x2 = (x + w / 2) * IMG_SIZE
                            y2 = (y + h / 2) * IMG_SIZE
                            ground_truths.append({
                                'image_id': img_idx,
                                'bbox': np.array([x1, y1, x2, y2])
                            })
        
        # Skip class if no ground truths
        if len(ground_truths) == 0:
            continue
            
        # Debug print
        print(f"Class {c}: {len(detections)} detections, {len(ground_truths)} ground truths")
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Initialize counters for precision/recall calculation
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        
        # Track which ground truths were used (avoid double counting)
        # For each image, create a set to track used ground truth boxes
        gt_used = {img_idx: set() for img_idx in range(len(true_boxes))}
        
        # Process each detection
        for d_idx, detection in enumerate(detections):
            img_id = detection['image_id']
            
            # Get ground truths for this image
            gt_this_img = [i for i, gt in enumerate(ground_truths) if gt['image_id'] == img_id]
            
            # No ground truths for this image
            if len(gt_this_img) == 0:
                FP[d_idx] = 1
                continue
                
            # Get detection bounding box
            d_bbox = detection['bbox']
            
            # Find best matching ground truth
            max_iou = -float('inf')
            max_gt_idx = -1
            
            for gt_idx in gt_this_img:
                # Skip if this ground truth was already used
                if gt_idx in gt_used[img_id]:
                    continue
                    
                # Calculate IoU
                gt_bbox = ground_truths[gt_idx]['bbox']
                iou = calculate_iou(d_bbox, gt_bbox)
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
                    
            # Check if detection matches any ground truth
            if max_iou >= iou_threshold and max_gt_idx != -1:
                # Mark this ground truth as used
                gt_used[img_id].add(max_gt_idx)
                TP[d_idx] = 1
            else:
                FP[d_idx] = 1
                
        # Calculate precision and recall
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum / len(ground_truths)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-16)  # Add small epsilon to avoid division by zero
        
        # Add sentinel values to start the precision-recall curve
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))
        
        # Calculate average precision
        AP = np.trapz(precisions, recalls)
        average_precisions.append(AP)
        print(f"Class {c} AP: {AP:.4f}")
    
    # Calculate mean AP
    if len(average_precisions) > 0:
        mAP = np.mean(average_precisions)
        print(f"Total valid predictions: {valid_predictions}/{total_predictions}")
        print(f"mAP: {mAP:.4f} (across {len(average_precisions)} classes)")
        return mAP
    else:
        print("No valid predictions or ground truths found!")
        return 0.0


# Helper function to calculate IoU for two boxes
def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in format [x1, y1, x2, y2].
    """
    # Determine coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0


# Set up data transformations
def get_transforms(train=True):
    """
    Get transformations for images.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(10),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Collate function for data loader
def collate_fn(batch):
    """
    Custom collate function for dataloader to handle variable sized targets.
    """
    images, targets = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Return images and targets
    return images, targets


# Training function
def train_model(model, train_loader, val_loader, epochs=5):
    """
    Improved training function with gradient clipping and better learning rate.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train for
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function
    criterion = YOLOLoss(ANCHORS, GRID_SIZES, NUM_CLASSES).to(device)
    
    # Optimizer with lower learning rate and weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # Lower initial learning rate 
        weight_decay=1e-4  # Add weight decay to prevent overfitting
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        # Warm up for first 3 epochs
        if epoch < 3:
            return (epoch + 1) / 3
        # Then follow cosine schedule
        return 0.5 * (1 + np.cos((epoch - 3) / (epochs - 3) * np.pi))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mAP': [],
        'learning_rates': []
    }
    
    # Start training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        batch_count = 0
        
        for images, targets in progress_bar:
            # Skip batch if all targets are empty
            if all(len(t) == 0 for t in targets):
                continue
                
            # Move to device
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Skip bad batches with extremely high loss
            if loss.item() > 1e6:
                print(f"Skipping batch with loss: {loss.item():.2e}")
                continue
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            
            optimizer.step()
            
            # Update progress bar
            batch_loss = loss.item() / images.size(0)  # Normalize by batch size
            epoch_loss += batch_loss
            batch_count += 1
            
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': batch_loss,
                'lr': lr
            })
        
        # Average training loss for the epoch
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

       # VALIDATION LOOP - update this part
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        # Store predictions and ground truth for mAP calculation
        all_pred_boxes = []
        all_true_boxes = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                # Skip batch if all targets are empty
                if all(len(t) == 0 for t in targets):
                    continue
                
                # Move to device
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                batch_loss = loss.item() / images.size(0)
                val_loss += batch_loss
                val_batch_count += 1
                
                # Process predictions
                for i in range(images.size(0)):
                    # Get raw predictions for this image
                    pred_boxes = []
                    
                    # Process each scale
                    for scale_idx, output in enumerate(outputs):
                        # Process this scale's predictions
                        batch, _, grid_size, _ = output.shape
                        num_anchors = len(ANCHORS[scale_idx])
                        num_classes = NUM_CLASSES
                        
                        # Reshape to [batch, anchors, grid, grid, 5+classes]
                        prediction = output.view(batch, num_anchors, 5+num_classes, grid_size, grid_size)
                        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
                        
                        # Apply sigmoid to confidence and class scores
                        prediction[..., 4:] = torch.sigmoid(prediction[..., 4:])
                        
                        # Process this image
                        img_pred = prediction[i]
                        
                        # Get boxes with confidence > threshold (0.1 for validation)
                        conf_mask = img_pred[..., 4] > 0.1
                        if not conf_mask.any():
                            continue
                            
                        # Get predicted boxes
                        pred = img_pred[conf_mask]
                        
                        # Get grid positions
                        grid_indices = conf_mask.nonzero()
                        anchor_idx, grid_y, grid_x = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]
                        
                        # Convert to absolute coordinates
                        boxes = torch.zeros((pred.size(0), 6), device=device)
                        boxes[:, 0] = (torch.sigmoid(pred[:, 0]) + grid_x) / grid_size * IMG_SIZE  # x center
                        boxes[:, 1] = (torch.sigmoid(pred[:, 1]) + grid_y) / grid_size * IMG_SIZE  # y center
                        
                        # Width and height
                        anchor_w = torch.tensor([a[0] for a in ANCHORS[scale_idx]], device=device)[anchor_idx]
                        anchor_h = torch.tensor([a[1] for a in ANCHORS[scale_idx]], device=device)[anchor_idx]
                        boxes[:, 2] = torch.exp(pred[:, 2]) * anchor_w * IMG_SIZE  # width
                        boxes[:, 3] = torch.exp(pred[:, 3]) * anchor_h * IMG_SIZE  # height
                        
                        # Convert to corner format
                        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
                        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
                        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
                        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2
                        
                        # Confidence and class
                        boxes[:, 4] = pred[:, 4]  # confidence
                        boxes[:, 5] = pred[:, 5:].argmax(1)  # class index
                        
                        pred_boxes.append(boxes)
                    
                    # Combine predictions from all scales
                    if pred_boxes:
                        img_boxes = torch.cat(pred_boxes, dim=0)
                        # Apply NMS
                        img_boxes = improved_nms(img_boxes, iou_threshold=0.5)
                        all_pred_boxes.append(img_boxes)
                    else:
                        all_pred_boxes.append(torch.zeros((0, 6), device=device))
                
                # Add targets to list - deep copy to avoid reference issues
                for target in targets:
                    all_true_boxes.append(target.clone())
        
        # Average validation loss
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        history['val_loss'].append(avg_val_loss)
        
        # Calculate mAP
        print("\nCalculating validation mAP...")
        if len(all_pred_boxes) > 0 and len(all_true_boxes) > 0:
            val_mAP = calculate_mAP(all_pred_boxes, all_true_boxes)
            history['val_mAP'].append(val_mAP)
        else:
            val_mAP = 0
            history['val_mAP'].append(0)
            print("No valid predictions or ground truths for mAP calculation!")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Print metrics
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  Val mAP: {val_mAP:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping if loss is NaN or too high
        if np.isnan(avg_train_loss) or avg_train_loss > 1e5:
            print("Stopping early due to unstable training")
            break
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mAP': val_mAP
        }, f'yolo_densenet_checkpoint_epoch{epoch+1}.pth')
    
    return model, history

# Visualization function for debugging
def visualize_predictions(model, image_path, conf_thresh=0.25):
    """
    Visualize predictions on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        conf_thresh: Confidence threshold for detections
    """
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    orig_img = np.array(image)
    
    # Apply transformations
    transform = get_transforms(train=False)
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        pred_boxes = process_predictions(outputs, ANCHORS, conf_thresh=conf_thresh)
        pred_boxes = improved_nms(pred_boxes)
    
    # Draw bounding boxes
    img_with_boxes = orig_img.copy()
    
    if pred_boxes.size(0) > 0:
        # Scale coordinates to original image size
        scale_x = orig_img.shape[1] / IMG_SIZE
        scale_y = orig_img.shape[0] / IMG_SIZE
        
        for box in pred_boxes:
            x1, y1, x2, y2 = box[:4]
            x1 = int(x1.item() * scale_x)
            y1 = int(y1.item() * scale_y)
            x2 = int(x2.item() * scale_x)
            y2 = int(y2.item() * scale_y)
            
            conf = box[4].item()
            cls_idx = int(box[5].item())
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{COCO_CLASSES[cls_idx]}: {conf:.2f}'
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.show()


# Demo function to detect objects in a test image
def demo(model_path, image_path, conf_thresh=0.25):
    """
    Demo function to load a trained model and detect objects in an image.
    
    Args:
        model_path: Path to trained model checkpoint
        image_path: Path to test image
        conf_thresh: Confidence threshold for detections
    """
    # Load model
    model = DenseNetYOLO(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Visualize predictions
    visualize_predictions(model, image_path, conf_thresh)


# Main function to run the training pipeline
def main():
    # Define paths to dataset
    coco_img_dir = '/kaggle/input/coco-2017-dataset/coco2017'
    coco_ann_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations'
    
    # Create datasets
    train_dataset = COCODataset(
        img_dir=coco_img_dir + '/train2017',
        annotations_file=coco_ann_file + '/instances_train2017.json',
        transform=get_transforms(train=True)
    )
    
    val_dataset = COCODataset(
        img_dir=coco_img_dir + '/val2017',
        annotations_file=coco_ann_file + '/instances_val2017.json',
        transform=get_transforms(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = DenseNetYOLO(num_classes=NUM_CLASSES)
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5  # Adjust as needed
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mAP'], label='Val mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Demo on a test image
    


if __name__ == '__main__':
    main()