import cv2
import mediapipe as mp


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
                eye_height = abs(
                    self._landmark_to_coords(left_eye_vertical[0])[1] - lower_lid[1]
                )  # Upper eyelid to bottom eyelid height
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

    def annotated_frame(self):
        """Returns the frame with eye landmarks, iris centers, and face mesh annotated."""
        frame = self.frame.copy()

        # Draw face mesh landmarks
        if self.frame_landmarks:
            h, w, _ = frame.shape
            points = []
            for landmark in self.frame_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

            # Draw lines connecting the face mesh points
            for start, end in self.mp_face_mesh.FACEMESH_TESSELATION:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 1)

        # Annotate iris centers
        if self.left_pupil:
            cv2.circle(frame, self.left_pupil, 3, (0, 0, 255), -1)
        if self.right_pupil:
            cv2.circle(frame, self.right_pupil, 3, (0, 255, 0), -1)

        normalized_distances = self.normalized_pupil_to_bottom_distance()
        if normalized_distances["left"] is not None:
            cv2.putText(frame, f"Left Eye Normalized: {normalized_distances['left']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if normalized_distances["right"] is not None:
            cv2.putText(frame, f"Right Eye Normalized: {normalized_distances['right']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame


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
        if self.calibration_values["up"] is not None and current_distance <= self.calibration_values["up"]:
            return "up"
        elif self.calibration_values["down"] is not None and current_distance >= self.calibration_values["down"]:
            return "down"
        elif self.calibration_values["center"] is not None:
            return "center"
        else:
            return "uncalibrated"