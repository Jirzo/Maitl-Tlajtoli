import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from settings.collect_image import load_font, DATA_DIR_ESTATIC, dataset_size, alphabet


def frame_instuctions():
    cap = cv2.VideoCapture(0)
    window_name = "Capture Images"
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    font = load_font()
    while True:
        print("\nSelect the letter to capture (A-Z). Enter '1' to exit.")
        selected_letter = input("Letter: ").strip().upper()

        if selected_letter == "1":
            break
        if selected_letter not in alphabet:
            print(
                "Invalid entry. Please select a letter (A-Z), the only letters that are not allowed are Ñ and LL."
            )
            continue

        class_dir = os.path.join(DATA_DIR_ESTATIC, selected_letter)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'\nPreparing capture for the letter "{selected_letter}"...')
        text = f'Capturing: {selected_letter} | Press "1" to stop'

        # Display text on screen until '1' is pressed
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not capture frame. Please check the camera.")
                break

            # Convert OpenCV frame to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            if font:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            else:
                text_width, text_height = 200, 30

            # Text position
            text_x, text_y = 20, 20
            padding = 10

            # Draw background rectangle
            rect_x0 = text_x - padding
            rect_y0 = text_y - padding
            rect_x1 = text_x + text_width + padding
            rect_y1 = text_y + text_height + padding
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 150))

            # Draw text
            text_color = (255, 255, 255)
            draw.text(
                (text_x, text_y),
                text,
                font=font or ImageFont.load_default(),
                fill=text_color,
            )

            # Convert PIL image to OpenCV
            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Show frame with text
            cv2.imshow(window_name, frame_with_text)

            # Close'Q'
            if cv2.waitKey(25) == ord("1"):
                break

        # Capture images for the selected letter
        print(f"Starting capture for '{selected_letter}'...")
        counter = 0

        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not capture frame during data collection.")
                break

            cv2.imshow(window_name, frame)
            cv2.waitKey(25)

            # Save image in the format <letter>_<number>.jpg
            filename = f"{selected_letter}_{counter:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, frame)
            counter += 1

            print(f"Image saved: {filename}")

        print(f"Screenshot for the full letter '{selected_letter}' .")

    cap.release()
    cv2.destroyAllWindows()
