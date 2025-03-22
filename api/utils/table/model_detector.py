import numpy as np
import torch
import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class InvoiceTableDetector:
    def __init__(self, model_name="Dilipan/detr-finetuned-invoice-item-table", conf_threshold=0.5):
        self.processor = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
    
    def detect_tables(self, image):
        """Detect tables in the invoice image and return the largest one."""
        # Convert image from BGR to RGB (Transformers model expects RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert detections to bounding boxes
        img_h, img_w, _ = image.shape
        results = []
        for logits, boxes in zip(outputs.logits, outputs.pred_boxes):
            scores = logits.softmax(-1)[..., :-1].max(-1).values  # Ignore 'no object' class
            keep = scores >= self.conf_threshold  # Apply confidence threshold
            
            for box, score in zip(boxes[keep], scores[keep]):
                x_min, y_min, x_max, y_max = (box * torch.tensor([img_w, img_h, img_w, img_h])).tolist()
                
                # Convert to integer and ensure bbox is within valid range
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                x_min, x_max = max(0, x_min), min(img_w, x_max)
                y_min, y_max = max(0, y_min), min(img_h, y_max)

                # Ensure bbox has valid width & height
                if x_max > x_min and y_max > y_min:
                    results.append({
                        "bbox": [x_min, y_min, x_max, y_max],
                        "score": float(score)
                    })

        # If no valid tables detected, return None
        if not results:
            return None
        
        # Select the table with the largest area
        largest_table = max(results, key=lambda r: (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1]))
        x_min, y_min, x_max, y_max = largest_table['bbox']
        
        # Crop the largest detected table
        cropped_table = image[y_min:y_max, x_min:x_max]
        return cropped_table if cropped_table.size > 0 else None