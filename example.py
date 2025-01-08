import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

calibration_phase = True
directions = ["up", "center", "down"]
current_direction_index = 0

while True:
    # Get a new frame from the webcam
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Perform calibration
    if calibration_phase:
        cv2.putText(
            frame,
            f"Look {directions[current_direction_index]} and press 'c'",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1)
        if key == ord("c"):
            gaze.calibrate_vertical(frame, directions[current_direction_index])
            current_direction_index += 1
            if current_direction_index >= len(directions):
                calibration_phase = False
                print("Calibration complete!")
        continue

    # Analyze the frame
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

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
    if calibrated_vertical_direction != "uncalibrated" and horizontal_direction:
        text = f"Looking {horizontal_direction} and {calibrated_vertical_direction}"
    elif calibrated_vertical_direction != "uncalibrated":
        text = f"Looking {calibrated_vertical_direction}"
    else:
        text = "Calibrate to detect gaze"

    # Annotate text
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display ratios if available
    if horizontal_ratio is not None:
        cv2.putText(frame, f"Horizontal Ratio: {horizontal_ratio:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Horizontal Ratio: N/A", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    vertical_ratio = gaze.vertical_ratio()
    if vertical_ratio is not None:
        cv2.putText(frame, f"Vertical Ratio: {vertical_ratio:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Vertical Ratio: N/A", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display calibrated vertical direction
    cv2.putText(
        frame,
        f"Calibrated Gaze: {calibrated_vertical_direction}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Display the frame
    cv2.imshow("Demo", frame)

    # Exit on pressing Escape
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
