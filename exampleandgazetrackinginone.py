import cv2
import mediapipe as mp
import time


class GazeTracking:
    """
    Tracks the user's gaze using MediaPipe Face Mesh.
    """

    def __init__(self):
        self.frame = None
        self.frame_landmarks = None
        self.left_pupil = None
        self.right_pupil = None
        self.calibration_values = {"up": None, "center": None, "down": None}

        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def refresh(self, frame):
        """Refreshes the frame and analyzes it."""
        self.frame = frame
        self._analyze()

    def _analyze(self):
        """Detects the face and extracts iris landmarks."""
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Store landmarks for the first detected face
            self.frame_landmarks = results.multi_face_landmarks[0]

            # Extract iris landmarks for left and right eyes
            self.left_pupil = self._get_iris_center(self.frame_landmarks, "left")
            self.right_pupil = self._get_iris_center(self.frame_landmarks, "right")
        else:
            self.frame_landmarks = None  # No landmarks detected
            self.left_pupil = None
            self.right_pupil = None

    def _get_iris_center(self, face_landmarks, side):
        """Calculates the geometric center of the iris."""
        LEFT_IRIS = [468, 469, 470, 471]
        RIGHT_IRIS = [473, 474, 475, 476]

        iris_indices = LEFT_IRIS if side == "left" else RIGHT_IRIS
        h, w, _ = self.frame.shape
        iris_coords = [
            (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
            for i in iris_indices
        ]

        if not iris_coords:
            return None

        # Calculate the center of the iris
        x_coords = [point[0] for point in iris_coords]
        y_coords = [point[1] for point in iris_coords]
        return int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords))

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 indicating horizontal gaze direction."""
        if self.left_pupil and self.frame_landmarks:
            left_eye_outer = [33, 133]  # Outer corners of the left eye
            outer_corner = self._landmark_to_coords(left_eye_outer[0])
            inner_corner = self._landmark_to_coords(left_eye_outer[1])

            if outer_corner and inner_corner:
                eye_width = abs(outer_corner[0] - inner_corner[0])
                if eye_width > 0:  # Avoid division by zero
                    return (self.left_pupil[0] - outer_corner[0]) / eye_width

        # Return None if landmarks are invalid or eye width is zero
        return None

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 indicating vertical gaze direction."""
        if self.left_pupil and self.frame_landmarks:
            left_eye_vertical = [159, 145]  # Upper and lower eyelid of the left eye
            lower_lid = self._landmark_to_coords(left_eye_vertical[1])  # Bottom eyelid
            iris_center = self.left_pupil  # Center of the iris

            if lower_lid:
                # Distance between iris center and bottom eyelid
                distance_to_bottom = abs(iris_center[1] - lower_lid[1])

                # Normalize distance by eye height
                upper_lid = self._landmark_to_coords(left_eye_vertical[0])  # Upper eyelid
                eye_height = abs(upper_lid[1] - lower_lid[1]) if upper_lid else 0
                if eye_height > 0:  # Avoid division by zero
                    normalized_distance = distance_to_bottom / eye_height

                    # Adjust ratio: Higher distance indicates "looking up"
                    return 1.0 - normalized_distance
        return None

    def _landmark_to_coords(self, index):
        """Converts a single MediaPipe landmark to pixel coordinates."""
        if self.frame_landmarks:
            h, w, _ = self.frame.shape
            landmark = self.frame_landmarks.landmark[index]
            return int(landmark.x * w), int(landmark.y * h)
        return None

    def calibrate_vertical(self, frame, gaze_direction):
        """
        Calibrates vertical gaze for a given direction ('up', 'center', 'down').
        """
        self.refresh(frame)
        normalized_distances = self.normalized_pupil_to_bottom_distance()
        if normalized_distances["left"] is not None:
            self.calibration_values[gaze_direction] = normalized_distances["left"]

    def get_calibrated_vertical_direction(self):
        """
        Determines gaze direction (up, center, down) based on calibrated thresholds.
        """
        normalized_distances = self.normalized_pupil_to_bottom_distance()
        if normalized_distances["left"] is None:
            return "no detection"

        current_distance = normalized_distances["left"]

        # Compare current distance with calibrated values
        if (
            self.calibration_values["up"] is not None
            and current_distance <= self.calibration_values["up"]
        ):
            return "up"
        elif (
            self.calibration_values["down"] is not None
            and current_distance >= self.calibration_values["down"]
        ):
            return "down"
        elif self.calibration_values["center"] is not None:
            # Check if current_distance is close to center calibration
            center = self.calibration_values["center"]
            # Define a threshold range around the center value
            threshold = 0.05  # 5% tolerance
            if center - threshold <= current_distance <= center + threshold:
                return "center"
            else:
                return "uncalibrated"
        else:
            return "uncalibrated"

    def pupil_to_bottom_distance(self):
        """
        Calculates the physical distance in pixels between the pupil center
        and the bottom eyelid for both eyes.
        Returns a dictionary with the distances for the left and right eyes.
        """
        distances = {"left": None, "right": None}

        if self.frame_landmarks:
            # Left eye
            left_eye_vertical = [159, 145]  # Upper and lower eyelid of the left eye
            lower_lid_left = self._landmark_to_coords(left_eye_vertical[1])  # Bottom eyelid
            if self.left_pupil and lower_lid_left:
                distances["left"] = abs(self.left_pupil[1] - lower_lid_left[1])

            # Right eye
            right_eye_vertical = [386, 374]  # Upper and lower eyelid of the right eye
            lower_lid_right = self._landmark_to_coords(right_eye_vertical[1])  # Bottom eyelid
            if self.right_pupil and lower_lid_right:
                distances["right"] = abs(self.right_pupil[1] - lower_lid_right[1])

        return distances

    def normalized_pupil_to_bottom_distance(self):
        """
        Calculates the normalized vertical distance between the pupil center
        and the bottom eyelid for both eyes, relative to the eye height.
        Returns a dictionary with the normalized distances for the left and right eyes.
        """
        distances = {"left": None, "right": None}

        if self.frame_landmarks:
            # Left eye
            left_eye_vertical = [159, 145]  # Upper and lower eyelid of the left eye
            upper_lid_left = self._landmark_to_coords(left_eye_vertical[0])  # Upper eyelid
            lower_lid_left = self._landmark_to_coords(left_eye_vertical[1])  # Bottom eyelid
            if self.left_pupil and upper_lid_left and lower_lid_left:
                eye_height_left = abs(upper_lid_left[1] - lower_lid_left[1])
                if eye_height_left > 0:  # Avoid division by zero
                    raw_distance_left = abs(self.left_pupil[1] - lower_lid_left[1])
                    distances["left"] = raw_distance_left / eye_height_left

            # Right eye
            right_eye_vertical = [386, 374]  # Upper and lower eyelid of the right eye
            upper_lid_right = self._landmark_to_coords(right_eye_vertical[0])  # Upper eyelid
            lower_lid_right = self._landmark_to_coords(right_eye_vertical[1])  # Bottom eyelid
            if self.right_pupil and upper_lid_right and lower_lid_right:
                eye_height_right = abs(upper_lid_right[1] - lower_lid_right[1])
                if eye_height_right > 0:  # Avoid division by zero
                    raw_distance_right = abs(self.right_pupil[1] - lower_lid_right[1])
                    distances["right"] = raw_distance_right / eye_height_right

        return distances


def main():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    calibration_phase = True
    directions = ["up", "center", "down"]
    current_direction_index = 0

    print("Starting webcam...")
    print("Calibration Phase: Please look in the specified direction and press 'c' to calibrate.")

    while True:
        # Get a new frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Flip the frame horizontally for a mirror effect (optional)
        frame = cv2.flip(frame, 1)

        # Perform calibration
        if calibration_phase:
            direction = directions[current_direction_index]
            print(f"Calibration {current_direction_index + 1}/{len(directions)}: Look {direction} and press 'c' to calibrate.")

            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                print(f"Calibrating for '{direction}' direction...")
                gaze.calibrate_vertical(frame, direction)
                print(f"Calibration for '{direction}' completed.\n")
                current_direction_index += 1
                if current_direction_index >= len(directions):
                    calibration_phase = False
                    print("All calibrations completed! Starting gaze tracking...\n")
                    cv2.destroyWindow("Calibration")
            elif key == 27:  # ESC key to exit
                print("Exiting calibration.")
                break
            continue

        # Analyze the frame
        gaze.refresh(frame)

        # Get horizontal and calibrated vertical gaze directions
        horizontal_ratio = gaze.horizontal_ratio()
        calibrated_vertical_direction = gaze.get_calibrated_vertical_direction()

        # Determine horizontal direction
        horizontal_direction = ""
        if horizontal_ratio is not None:
            if horizontal_ratio <= 0.35:
                horizontal_direction = "right"
            elif horizontal_ratio >= 0.65:
                horizontal_direction = "left"
            else:
                horizontal_direction = "center horizontally"

        # Combine horizontal and calibrated vertical directions
        if calibrated_vertical_direction not in ["uncalibrated", "no detection"]:
            if horizontal_direction:
                gaze_direction = f"Looking {horizontal_direction} and {calibrated_vertical_direction}"
            else:
                gaze_direction = f"Looking {calibrated_vertical_direction}"
        else:
            gaze_direction = "Calibrate to detect gaze"

        # Get normalized ratios
        normalized_distances = gaze.normalized_pupil_to_bottom_distance()
        horizontal_ratio_str = f"{horizontal_ratio:.2f}" if horizontal_ratio is not None else "N/A"
        vertical_ratio = gaze.vertical_ratio()
        vertical_ratio_str = f"{vertical_ratio:.2f}" if vertical_ratio is not None else "N/A"

        # Print the statistics to the terminal
        print(f"Gaze Direction: {gaze_direction}")
        print(f"Horizontal Ratio: {horizontal_ratio_str} | Vertical Ratio: {vertical_ratio_str}")
        print(f"Calibrated Gaze: {calibrated_vertical_direction}")
        print("-" * 50)

        # **Print Landmark 468 Data**
        if gaze.frame_landmarks:
            landmark468 = gaze.frame_landmarks.landmark[468]
            print(f"Landmark 468 - x: {landmark468.x:.4f}, y: {landmark468.y:.4f}, z: {landmark468.z:.4f}")
        else:
            print("Landmark 468: No detection")
        print("=" * 50)  # Separator for landmark data

        # Display the frame without annotations
        cv2.imshow("Gaze Tracking", frame)

        # Exit on pressing Escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("Exiting application.")
            break

        # To reduce terminal flooding, you can add a short delay
        time.sleep(0.1)  # Adjust as needed

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
