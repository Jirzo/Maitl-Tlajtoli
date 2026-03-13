import cv2
import os

# --- CONFIGURATION---
DATA_DIR = "data/data_dinamico"
N_FRAMES = 30  # The window size we configured earlier


def prepare_directory(clase):
    """
    It creates the folder if it doesn't exist and calculates the frame number where we left off.
    It runs ONLY when you change the signal to avoid locking the camera.
    """
    path = os.path.join(DATA_DIR, clase)
    os.makedirs(path, exist_ok=True)

    existing_files = os.listdir(path)
    if existing_files:
        numbers = [
            int(f.split("_")[1].split(".")[0])
            for f in existing_files
            if f.startswith("frame_")
        ]
        counter = max(numbers) + 1 if numbers else 0
    else:
        counter = 0

    return path, counter


# --- PROGRAM START ---
signal_class = (
    input("What dynamic signal are you going to record? (ej. J, Z, verbo_comer): ")
    .strip()
    .upper()
)

# We prepared the folder for the first time
class_path, frame_counter = prepare_directory(signal_class)

# Initialize the webcam
cap = cv2.VideoCapture(0)

print(f"\n[INFO] Ready to record the signal: {signal_class}")
print("[INFO] Press 'r' to record a sequence of 30 frames.")
print("[INFO] Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # We draw text on the screen
    cv2.putText(
        frame,
        f"Sena: {signal_class} | Saved frames: {frame_counter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Press 'R' to record, 'Q' to exit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Dynamic Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == ord("q"):
        break

    # Start burst recording
    elif key == ord("r"):
        print(f"\n[REC] Recording a sequence of 30 frames for {signal_class}...")
        captured_frames = 0

        while captured_frames < N_FRAMES:
            ret, frame_rec = cap.read()
            if not ret:
                break

            file_name = f"frame_{frame_counter:04d}.jpg"
            file_path = os.path.join(class_path, file_name)

            cv2.imwrite(file_path, frame_rec)

            cv2.putText(
                frame_rec,
                f"RECORDING: {captured_frames + 1}/{N_FRAMES}",
                (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

            cv2.imshow("Dynamic Collector", frame_rec)
            cv2.waitKey(1)

            frame_counter += 1
            captured_frames += 1

        print("[OK] Sequence saved successfully.")

        # --- SIGNAL CHANGE ---
        # We paused the program in the terminal to ask if we changed the sign-in.
        print("\nIf you want to continue using the same signal, just press Enter..")
        new_signal = (
            input("If you want to record another signal, write it here.: ")
            .strip()
            .upper()
        )

        # Si el usuario escribió algo nuevo, actualizamos la ruta y el contador
        if new_signal != "":
            signal_class = new_signal
            class_path, frame_counter = prepare_directory(signal_class)
            print(f"\n[INFO] Cambiando a seña: {signal_class}")

cap.release()
cv2.destroyAllWindows()
