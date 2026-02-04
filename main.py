from dataset.create_dataset import datasetcCreation
from dataset.create_dataset_tf import datasetTFCreation_LSTM
from hands_detection.landmarks import mediapipe_detection_fn
from image_collection.collect_img import frame_instuctions
from training.randomForestTrainer import randomForestClassifier
from inference_classifier.classifier import iClassifier
from tensorflowModel.tensorModel import tf_trainer
from settings.landmarks import mp_hands

hands_model = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
)

# Options menu
print("Select the action you want to perform:")
print("1: Collecting images")
print("2: Create dataset")
print("3: Training a model")
print("4: Classifier")

# Windows title
window_title = ""

try:
    option_selected = int(input("Select an option (1-4): "))
    if option_selected not in [1, 2, 3, 4]:
        raise ValueError("Invalid option")
except ValueError:
    print("Error: You must enter a number between 1 and 4.")
    exit()

# Ejecución de la opción seleccionada
if option_selected == 1:
    print("Starting image collection...")
    window_title = "Collect Images"
    frame_instuctions()
    exit()

elif option_selected == 2:
    print("Creating dataset...")
    datasetTFCreation_LSTM(hands_model)
    exit()

elif option_selected == 3:
    print("Training model...")
    tf_trainer()
    exit()

elif option_selected == 4:
    print("Running classifier inference...")
    iClassifier(hands_model)
