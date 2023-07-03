import argparse

import cv2
import numpy as np
import pytesseract


def preprocess_image(image):
    # other preprocessing should be done if needed
    return image


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated_image


def fix_orientation(image):
    # orientation and script detection
    osd = pytesseract.image_to_osd(image)

    angle = 0
    for line in osd.split("\n"):
        if line.startswith("Rotate:"):
            angle = int(line.split(": ")[-1])

    image = rotate_image(image, -angle)

    return image


def perform_ocr(image, language, not_horizontal, psm, format):
    config = ""
    if not_horizontal:
        image = fix_orientation(image)
    if psm:
        psm_values = [0, 3, 6, 11, 13]
        text = ""
        for psm_value in psm_values:
            config = f"--psm {psm_value}"
            extracted_text = pytesseract.image_to_string(
                image, lang=language, config=config
            )
            text += f"PSM={psm_value}:\n{extracted_text}\n"
    else:
        if format == "json":
            text = pytesseract.image_to_string(
                image,
                lang=language,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        else:
            text = pytesseract.image_to_string(
                image,
                lang=language,
                config=config,
            )
    return text


def main():
    parser = argparse.ArgumentParser(description="Pytesseract OCR Experiments")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument(
        "--language",
        type=str,
        default="eng",
        help="Language for OCR (default: eng)",
    )
    parser.add_argument(
        "--not-horizontal",
        action="store_true",
        help="Extract vertical text",
    )
    parser.add_argument(
        "--page-segmentation-mode",
        action="store_true",
        help="psm",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for OCR results (default: text)",
    )
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        image = preprocess_image(image)

        text = perform_ocr(
            image,
            args.language,
            args.not_horizontal,
            args.page_segmentation_mode,
            args.format,
        )

        print("Extracted Text:")
        print(text)


if __name__ == "__main__":
    main()
