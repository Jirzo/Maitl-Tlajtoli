from dataset.create_dataset_tf_GF import datasetTFCreation_Static
from dataset.create_dataset_tf_D import datasetTFCreation_Dynamic
from hands_detection.landmarks import mediapipe_detection_fn
from image_collection.collect_img import frame_instuctions
from training.randomForestTrainer import randomForestClassifier
from inference_classifier.classifier import scanner_tlajtoli
from tensorflowModel.tensorModel_LSTM_Dynamic import tf_trainer
from tensorflowModel.tensorData_Dense_Static import tf_static_trainer
from settings.landmarks import mp_hands

# Options menu
print("Select the action you want to perform:")
print("1: Collecting images")
print("2: Create dataset with static features")
print("3: Create dataset with dynamic features")
print("4: Training a Dynamic Model")
print("5: Training a Static Model")
print("6: Classifier")


def get_hands_model():
    print("Initializing MediaPipe Hands...")
    return mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    )


# Windows title
window_title = ""

try:
    option_selected = int(input("Select an option (1-6): "))
    if option_selected not in [1, 2, 3, 4, 5, 6]:
        raise ValueError("Invalid option")
except ValueError:
    print("Error: You must enter a number between 1 and 6.")
    exit()


# Ejecución de la opción seleccionada
if option_selected == 1:
    print("Starting image collection...")
    window_title = "Collect Images"
    frame_instuctions()
    exit()

elif option_selected == 2:
    print("Creating dataset with static features...")
    hm = get_hands_model()
    datasetTFCreation_Static(hm)
    exit()

elif option_selected == 3:
    print("Creating dataset with dynamic features...")
    hm = get_hands_model()
    datasetTFCreation_Dynamic(hm)
    exit()

elif option_selected == 4:
    print("Training Dynamic Model...")
    tf_trainer()
    exit()

elif option_selected == 5:
    print("Training Static Model...")
    tf_static_trainer()
    exit()

elif option_selected == 6:
    print("Running classifier inference...")
    hm = get_hands_model()
    scanner_tlajtoli()
