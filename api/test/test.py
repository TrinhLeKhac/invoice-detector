import unittest
import cv2
import numpy as np
import base64

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image import (
    convert_from_base64, convert_to_base64, get_rotate_angle,
    rotate_image, process_image, show_image, 
    extract_information_from_image
)
from utils.text import process_output

class TestImageProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # cls.image_path = str(Path(__file__).parent / "data/skew-linedetection.png")
        cls.image_path = str(Path(__file__).parent / "data/image015.jpg")
        cls.image = cv2.imread(cls.image_path)

        if cls.image is None:
            raise FileNotFoundError(f"Could not read image at path: {cls.image_path}")

    def test_convert_from_base64(self):
        base64_str = convert_to_base64(self.image)
        # print(base64_str)

        decoded_image = convert_from_base64(base64_str)
        show_image("Original image", self.image)
        show_image("Decoded image", decoded_image)

        self.assertIsNotNone(decoded_image)
        self.assertEqual(decoded_image.shape, self.image.shape)
    
    def test_convert_to_base64(self):
        base64_str = convert_to_base64(self.image)
        self.assertIsInstance(base64_str, str)
        self.assertGreater(len(base64_str), 0)
    
    def test_get_rotate_angle(self):
        angle = get_rotate_angle(self.image)
        print(f'Rotation: {angle:.2f} degrees')
        self.assertIsInstance(angle, (int, float))
    
    def test_rotate_image(self):
        rotated = rotate_image(self.image)
        self.assertEqual(rotated.shape, self.image.shape)
    
    def test_process_image(self):
        gray, rotated, binary, cropped = process_image(self.image)
        self.assertIsNotNone(gray)
        self.assertIsNotNone(rotated)
        self.assertIsNotNone(binary)
        self.assertIsNotNone(cropped)

        show_image("Gray image", gray)
        show_image("Rotated image", rotated)
        show_image("Binary image", binary)
        show_image("Cropped image", cropped)
    
    def test_extract_information_from_image(self):
        extracted_text = extract_information_from_image(self.image, config="--psm 6", lang="eng")
        # print(extracted_text)
        self.assertIsInstance(extracted_text, str)

    def test_process_output(self):
        extracted_text = extract_information_from_image(self.image, config="--psm 6", lang="eng")
        print(extracted_text)
        profile_info, order_details, order_summary = process_output(extracted_text)
        for k, v in profile_info.items():
            print(k, v)

if __name__ == "__main__":
    unittest.main()
