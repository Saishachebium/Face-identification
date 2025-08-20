
# Face Identification

![MIT License](https://img.shields.io/badge/license-MIT-green)  

This project identifies whether a user’s face matches an existing record using computer vision techniques. It leverages the Haar Cascade classifier for face detection and ORB (Oriented FAST and Rotated BRIEF) features for facial recognition. The program compares extracted facial features with stored biometric data to determine a match.

## Features

- Real-time face detection using Haar Cascade classifier.  
- ORB feature extraction for reliable matching of facial keypoints.  
- Duplicate image detection to avoid storing the same face multiple times.  
- Simple login/sign-up workflow for registering and verifying faces.  
- Handles real-world factors like lighting, face positioning, and camera quality.  

## How It Works

1. The program accesses the webcam to capture live video frames.  
2. It uses a Haar Cascade classifier to detect faces in each frame.  
3. The detected face is cropped, converted to grayscale, and normalized for consistency.  
4. ORB features are extracted from the normalized face image.  
5. These features are compared with stored biometric data to determine if there is a match.  
6. If a match is found, the program welcomes the user. If no match is found, the user can register by providing their name, and their facial features are stored for future recognition.

## Factors Affecting Accuracy

The reliability of face identification depends on several factors:

- **Camera quality** – Higher resolution images improve feature detection.  
- **Lighting conditions** – Shadows, glare, or low light can distort facial features.  
- **Face positioning** – Rotation, tilting, or movement may reduce matching accuracy.  
- **Feature clarity** – Blurry or partially visible faces can affect recognition results.  

## Usage

### Sign Up / Login

1. Run the program. The webcam window will appear.  
2. Press **1** to start the login/sign-up process.  
3. If you are a new user:  
   - Position your face in the camera view.  
   - Ensure your face is clearly visible and well-lit.  
   - Enter your name when prompted to register.  
4. If you are an existing user:  
   - Position your face in the camera view.  
   - Wait for the program to determine if your face matches a stored record.  

> **Note:** Face recognition is **not 100% accurate**. Factors such as lighting, camera quality, and face positioning can affect results.

## Future Improvements

- Enhance matching accuracy using multiple images per user.  
- Add support for multiple simultaneous users.  
- Improve feature detection with more advanced algorithms like SIFT or deep learning models.  
- Develop a GUI for easier interaction.  

## License

This project is licensed under the MIT License.
