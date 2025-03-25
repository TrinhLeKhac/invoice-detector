import os

import cv2
import torch
from PIL import Image
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from torchvision import transforms
from transformers import AutoModelForObjectDetection


class TableDetector(object):
    """
    Class for detecting tables in images without saving temporary images.
    Takes input images as PIL objects or numpy arrays and returns processed images.
    """

    _model = None  # Static variable to store the table detection model
    _device = None  # Static variable to store device information

    class MaxResize:
        """
        Helper class to resize images while maintaining aspect ratio
        and ensuring they do not exceed a maximum size.
        """

        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            height, width = image.shape[:2]
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = cv2.resize(
                image, (int(round(scale * width)), int(round(scale * height)))
            )
            return resized_image

    @classmethod
    def _initialize_model(cls, invoke_pipeline_step, local):
        """
        Initialize the table detection model if not already loaded.
        """
        if cls._model is None:
            cls._model, cls._device = invoke_pipeline_step(
                lambda: cls.load_table_detection_model(),
                "Loading table detection model...",
                local,
            )
            print("Table detection model initialized.")

    def detect_tables(self, image, local=True, debug=False):
        """
        Detect tables in an image and return cropped table images.

        :param image: Input image in PIL or numpy array format.
        :param local: Whether to show progress during processing.
        :param debug: Whether to display debug information.
        :return: List of cropped table images.
        """
        self._initialize_model(self.invoke_pipeline_step, local)
        model, device = self._model, self._device

        outputs, image = self.invoke_pipeline_step(
            lambda: self.prepare_image(image, model, device),
            "Preparing image for table detection...",
            local,
        )

        objects = self.invoke_pipeline_step(
            lambda: self.identify_tables(model, outputs, image.shape[:2]),
            "Identifying tables in the image...",
            local,
        )

        cropped_tables = self.invoke_pipeline_step(
            lambda: self.crop_tables(image, objects, debug),
            "Cropping tables from image...",
            local,
        )

        return cropped_tables

    @staticmethod
    def load_table_detection_model():
        """
        Load the table detection model.
        """
        model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, device

    def prepare_image(self, image, model, device):
        """
        Convert input OpenCV image into a tensor suitable for the model.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        image = self.MaxResize(800)(image)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )(image_tensor)
        pixel_values = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        return outputs, image

    def identify_tables(self, model, outputs, img_size):
        """
        Identify tables from model output.
        """
        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"
        return self.outputs_to_objects(outputs, img_size, id2label)

    def crop_tables(self, image, objects, debug):
        """
        Crop tables from the image based on detection results.
        """
        detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10,
        }
        crop_padding = 30

        tables_crops = self.objects_to_crops(
            image, [], objects, detection_class_thresholds, padding=crop_padding
        )

        if not tables_crops:
            if debug:
                print("No tables detected.")
            return None

        largest_crop = max(
            tables_crops,
            key=lambda crop: crop["image"].shape[0] * crop["image"].shape[1],
        )

        return largest_crop["image"]

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """
        Converts bounding box coordinates from (x_center, y_center, width, height)
        format to (x_min, y_min, x_max, y_max) format.

        Args:
            x (Tensor): A tensor of shape (N, 4), where each row contains
                        [x_center, y_center, width, height].

        Returns:
            Tensor: Bounding boxes converted to the format [x_min, y_min, x_max, y_max].
        """
        x_c, y_c, w, h = x.unbind(
            -1
        )  # Split tensor into four components: x_center, y_center, width, height
        b = [
            (x_c - 0.5 * w),  # x_min = x_center - (width / 2)
            (y_c - 0.5 * h),  # y_min = y_center - (height / 2)
            (x_c + 0.5 * w),  # x_max = x_center + (width / 2)
            (y_c + 0.5 * h),  # y_max = y_center + (height / 2)
        ]
        return torch.stack(b, dim=1)  # Stack into a tensor of shape (N, 4)

    def rescale_bboxes(self, out_bbox, size):
        """
        Converts normalized bounding box coordinates back to the original image size.

        Args:
            out_bbox (Tensor): Bounding boxes in normalized format (cxcywh with values in range [0,1]).
            size (tuple): Original image size (width, height).

        Returns:
            Tensor: Bounding boxes scaled to the original image size.
        """
        img_w, img_h = size  # Get image width and height
        b = self.box_cxcywh_to_xyxy(out_bbox)  # Convert from cxcywh to xyxy
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        """
        Converts model outputs into a list of bounding boxes and object labels.

        Args:
            outputs (dict): Model output containing detection results.
            img_size (tuple): Original image size (width, height).
            id2label (dict): Mapping from label ID to class name.

        Returns:
            list: A list of detected objects with bounding boxes and labels.
        """
        # Get the highest probability class label from the logits
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]  # Predicted class IDs
        pred_scores = list(m.values.detach().cpu().numpy())[
            0
        ]  # Corresponding confidence scores
        pred_bboxes = (
            outputs["pred_boxes"].detach().cpu()[0]
        )  # Extract bounding boxes from output
        pred_bboxes = [
            elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)
        ]

        # Create a list of detected objects
        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if class_label != "no object":  # Ignore invalid objects
                objects.append(
                    {"label": class_label, "score": float(score), "bbox": bbox}
                )

        return objects

    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=10):
        """
        Extracts table crops from an image based on detected bounding boxes.

        Args:
            img (PIL.Image): The original image.
            tokens (list): List of OCR tokens (if available).
            objects (list): List of detected tables.
            class_thresholds (dict): Confidence thresholds for accepting tables.
            padding (int): Padding around the detected table.

        Returns:
            list: A list of cropped table images and associated token information.
        """
        table_crops = []

        for obj in objects:
            if (
                obj["score"] < class_thresholds[obj["label"]]
            ):  # Skip tables with low confidence
                continue

            bbox = obj["bbox"]
            x_min = max(0, int(bbox[0] - padding))
            y_min = max(0, int(bbox[1] - padding))
            x_max = min(img.shape[1], int(bbox[2] + padding))
            y_max = min(img.shape[0], int(bbox[3] + padding))

            cropped_img = img[y_min:y_max, x_min:x_max].copy()

            # Filter tokens that belong to the table
            table_tokens = [
                token for token in tokens if self.iob(token["bbox"], bbox) >= 0.5
            ]
            for token in table_tokens:
                token["bbox"] = [
                    token["bbox"][0] - x_min,
                    token["bbox"][1] - y_min,
                    token["bbox"][2] - x_min,
                    token["bbox"][3] - y_min,
                ]

            # Handle rotated tables
            if obj["label"] == "table rotated":
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                for token in table_tokens:
                    bbox = token["bbox"]
                    bbox = [
                        cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2],
                    ]
                    token["bbox"] = bbox

            table_crops.append({"image": cropped_img, "tokens": table_tokens})

        return table_crops

    @staticmethod
    def iob(boxA, boxB):
        """
        Computes the Intersection over Box (IoB) ratio between two bounding boxes.

        Args:
            boxA (list): [x_min, y_min, x_max, y_max] coordinates of box A.
            boxB (list): [x_min, y_min, x_max, y_max] coordinates of box B.

        Returns:
            float: IoB value in the range [0,1].
        """
        # Determine the intersection area
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

        # Compute intersection area
        inter_width = max(0, xB - xA + 1)
        inter_height = max(0, yB - yA + 1)
        interArea = inter_width * inter_height

        # Compute area of box A
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

        # Avoid division by zero
        if boxAArea == 0:
            return 0.0

        return interArea / boxAArea

    @staticmethod
    def invoke_pipeline_step(task_call, task_description, local):
        """
        Executes a processing step in the pipeline with progress indication.

        Args:
            task_call (function): The function representing the task to execute.
            task_description (str): Description of the task.
            local (bool): If True, display a progress bar.

        Returns:
            Any: The result returned from task_call().
        """
        if local:
            with Progress(
                SpinnerColumn(),  # Show a spinning indicator
                TextColumn(
                    "[progress.description]{task.description}"
                ),  # Display task description
                transient=False,  # Keep the progress bar visible after completion
            ) as progress:
                progress.add_task(
                    description=task_description, total=None
                )  # Add task to progress bar
                result = task_call()  # Execute the task
        else:
            print(task_description)  # Print task description if not running locally
            result = task_call()

        return result
