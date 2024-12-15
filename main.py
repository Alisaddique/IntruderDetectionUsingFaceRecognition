import cv2
import face_recognition
import numpy as np
import glob
import os
import time
import threading
import logging
import dropbox
from twilio.rest import Client
import uuid  # For generating unique filenames

# Configurations
KNOWN_IMAGES_PATH = 'C:\\Users\\itsal\\OneDrive\\Desktop\\Client Project\\knownimages\\*'
INTRUDERS_DIRECTORY = 'C:\\Users\\itsal\\OneDrive\\Desktop\\Client Project\\intruders\\'
CAMERA_INDEX = 0  # Change to your camera index if different
THRESHOLD = 0.65  # Face matching threshold
COOLDOWN_PERIOD = 10  # Cooldown time (seconds) for intruder detection
MASK_THRESHOLD = 0.3  # Adjust this for masked detection
# Twilio WhatsApp API credentials
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
WHATSAPP_FROM = 'whatsapp:+14155238886'  # Replace with Twilio WhatsApp-enabled number
WHATSAPP_TO = 'whatsapp:+923190855540'  # Replace with your WhatsApp number

# Dropbox API credentials
DROPBOX_ACCESS_TOKEN = ''
DROPBOX_UPLOAD_DIR = '/IDSMSEE/'  # Dropbox directory for intruder images

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread-safe dictionary to track intruder cooldown
intruder_face_cache = {}
intruder_faces_lock = threading.Lock()

# Initialize background subtraction for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Twilio client for WhatsApp
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


# Function to send WhatsApp alerts
def send_whatsapp_alert(image_url):
    message_body = f"Intruder detected! See image: {image_url}"
    message = client.messages.create(
        body=message_body,
        from_=WHATSAPP_FROM,
        to=WHATSAPP_TO
    )
    logging.info(f"WhatsApp alert sent: {message.sid}")


# Function to upload intruder image to Dropbox
def upload_to_dropbox(image_path):
    with open(image_path, 'rb') as f:
        file_name = os.path.basename(image_path)
        dropbox_path = DROPBOX_UPLOAD_DIR + file_name

        try:
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.add)  # Use add mode for safety
            logging.info(f"File uploaded to Dropbox: {dropbox_path}")
        except dropbox.exceptions.ApiError as e:
            logging.error(f"Error uploading to Dropbox: {e}")
            return None

        # Try to get an existing shared link or create a new one
        try:
            shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
            image_url = shared_link_metadata.url
            logging.info(f"Shared link created: {image_url}")
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error,
                          dropbox.sharing.CreateSharedLinkWithSettingsError) and e.error.is_shared_link_already_exists():
                logging.info(f"Shared link already exists for {file_name}. Retrieving the existing link.")
                shared_links = dbx.sharing_list_shared_links(path=dropbox_path).links
                if shared_links:
                    image_url = shared_links[0].url
                    logging.info(f"Reused existing shared link: {image_url}")
                else:
                    logging.error(f"No shared link found for {dropbox_path}.")
                    return None
            else:
                raise e  # Re-raise if it's an unhandled error

        return image_url


# Function to augment images
def augment_image(image):
    augmented_images = [image]
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    return augmented_images


# Load known images and their encodings
def load_known_faces():
    logging.info("Loading and encoding known images...")
    paths = glob.glob(KNOWN_IMAGES_PATH)
    image_encodings = []
    image_names = []

    for img_path in paths:
        logging.info(f"Processing image: {img_path}")
        try:
            img = face_recognition.load_image_file(img_path)
            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                encodings = face_recognition.face_encodings(aug_img)
                if len(encodings) > 0:
                    image_encodings.append(encodings[0])
                    image_names.append(os.path.basename(img_path).split('.')[0])
                else:
                    logging.warning(f"No face found in image: {img_path}")
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")

    logging.info(f"Loaded {len(image_encodings)} face encodings.")
    return image_encodings, image_names


# Check if face is partially masked
def is_face_masked(face_landmarks):
    if 'nose_bridge' in face_landmarks and len(face_landmarks['nose_bridge']) > 0:
        return False  # Face not masked
    return True  # Face may be masked


# Intruder detection thread
def process_frame(frame, known_encodings, known_names, intruder_face_cache, count):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings,
                                                                             face_landmarks_list):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < THRESHOLD:
                name = known_names[best_match_index]

            if is_face_masked(face_landmarks):
                name = f"Masked {name}" if name != 'Unknown' else 'Unknown Masked'
                logging.info(f"Detected possible masked person: {name}")

            if name == 'Unknown' or 'Masked' in name:
                current_time = time.time()

                with intruder_faces_lock:
                    intruder_found = False
                    for past_face_encoding, timestamp in intruder_face_cache.items():
                        if face_recognition.compare_faces([past_face_encoding], face_encoding, tolerance=0.6)[0]:
                            intruder_found = True
                            if current_time - timestamp < COOLDOWN_PERIOD:
                                logging.info("Intruder detected recently, skipping.")
                                break

                    if not intruder_found:
                        # Use a UUID to generate a unique filename for each intruder
                        unique_id = str(uuid.uuid4())
                        intruder_image_path = os.path.join(INTRUDERS_DIRECTORY, f'intru-{unique_id}.jpg')
                        cv2.imwrite(intruder_image_path, frame)
                        logging.info(f"Intruder saved at: {intruder_image_path}")
                        intruder_face_cache[tuple(face_encoding)] = current_time

                        # Upload to Dropbox and send WhatsApp alert
                        image_url = upload_to_dropbox(intruder_image_path)
                        if image_url:
                            send_whatsapp_alert(image_url)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame, count


# Function to detect motion
def detect_motion(frame):
    fg_mask = bg_subtractor.apply(frame)
    motion_detected = np.sum(fg_mask) > 1e5
    return motion_detected, fg_mask


# Main system function
def intruder_detection_system():
    if not os.path.exists(INTRUDERS_DIRECTORY):
        os.makedirs(INTRUDERS_DIRECTORY)

    known_encodings, known_names = load_known_faces()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    count = 0

    def capture_and_process():
        nonlocal count
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            frame_resized = cv2.resize(frame, (640, 480))

            motion_detected, fg_mask = detect_motion(frame_resized)
            if motion_detected:
                logging.info("Motion detected, processing frame for intruders...")
                frame_resized, count = process_frame(frame_resized, known_encodings, known_names, intruder_face_cache,
                                                     count)

            cv2.imshow('Intruder Detection', frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    try:
        capture_and_process()
    except Exception as e:
        logging.error(f"Error in detection system: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if _name_ == "_main_":
    intruder_detection_system()