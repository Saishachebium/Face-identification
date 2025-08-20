import cv2
import os
import pickle
import hashlib

# Create necessary directories
os.makedirs("./captured_Image/", exist_ok=True)
os.makedirs("output_directory", exist_ok=True)

# Load the Haar Cascade for face detection
face_detector_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(face_detector_path)
if face_detector.empty():
    raise RuntimeError(f"Failed to load Haar Cascade XML from {face_detector_path}")

# Initialize the ORB detector
orb = cv2.ORB_create(1000)  # Increased features for better matching

# Initialize biometrics dictionary and load previous data if available
Biometrics = {}
if os.path.exists("biometrics.pkl"):
    try:
        with open("biometrics.pkl", "rb") as f:
            Biometrics = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print("Error loading biometrics file. Starting with an empty database.")
else:
    print("Biometrics file not found. Starting fresh.")

# Initialize image counter and load previous value if available
image_counter_path = "image_counter.pkl"
if os.path.exists(image_counter_path):
    with open(image_counter_path, "rb") as f:
        image_counter = pickle.load(f)
else:
    image_counter = 0

# Function to save biometrics data
def save_biometrics():
    with open("biometrics.pkl", "wb") as f:
        pickle.dump(Biometrics, f)

# Function to save the image counter
def save_image_counter():
    with open(image_counter_path, "wb") as f:
        pickle.dump(image_counter, f)

# Function to convert ORB keypoints to a picklable format
def convert_keypoints(keypoints):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints]

# Normalize the cropped face image
def preprocess_face(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_face = cv2.resize(gray_face, (150, 150))  # Resize for consistency
    return resized_face

# Function to calculate a hash for an image
def calculate_image_hash(image):
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    return hashlib.md5(image_bytes).hexdigest()

# Function to compare features for a better match
def match_features(descriptors, stored_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, stored_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    # Adjust similarity threshold as needed
    similarity_threshold = 50  # Lower distance means higher similarity
    similar_features = [m for m in matches if m.distance < similarity_threshold]

    return len(similar_features)

# Function to capture an image and avoid duplicates
def capture_image(frame, path):
    image_hash = calculate_image_hash(frame)
    for existing_file in os.listdir("./captured_Image/"):
        existing_image_path = os.path.join("./captured_Image/", existing_file)
        existing_image = cv2.imread(existing_image_path)
        if existing_image is not None and calculate_image_hash(existing_image) == image_hash:
            print("Duplicate image detected. Skipping save.")
            return False

    cv2.imwrite(path, frame)
    print(f"Image saved at {path}")
    return True

# Start the camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Failed to access the camera. Ensure it is connected and not in use by another program.")

print("Press '1' to login or check biometrics. Press 'Esc' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.putText(frame, "Press '1' to login or sign up. Press 'Esc' to exit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):  # Trigger login/signup process
        print("Key '1' pressed")

        # Generate unique image path
        image_path = f"./captured_Image/current_image_{image_counter}.jpg"
        if capture_image(frame, image_path):
            # Increment and save the image counter only if a new image was saved
            image_counter += 1
            save_image_counter()

        # Convert the image to grayscale and detect faces
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            print("Face detected")
            (x, y, w, h) = faces[0]  # Use the first detected face
            cropped_face = frame[y:y + h, x:x + w]

            # Normalize the face for better matching
            normalized_face = preprocess_face(cropped_face)

            # Detect ORB features
            keypoints, descriptors = orb.detectAndCompute(normalized_face, None)

            if descriptors is not None:
                # Compare the face with stored biometrics
                found_match = False
                for name, (stored_keypoints, stored_descriptors) in Biometrics.items():
                    num_similar_features = match_features(descriptors, stored_descriptors)

                    if num_similar_features > 5:  # Lower threshold for a close match
                        print(f"Match found! Welcome, {name}.")
                        found_match = True
                        break

                if not found_match:
                    print("No match found. Adding you as a new user.")
                    name = input("Enter your name: ").strip()
                    if name:
                        Biometrics[name] = (convert_keypoints(keypoints), descriptors)
                        save_biometrics()
                        print(f"Biometric data saved for {name}.")
            else:
                print("No features detected. Please try again.")

        else:
            print("No face detected. Please try again.")

    elif key == 27:  # Esc key to exit
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
