import os
import csv
import cv2
import serial
import time
import threading

from datetime import datetime
from ultralytics import YOLO


class SecurityEngine:
    def __init__(
        self,
        arduino_port="COM12",
        model_path="yolo11n.pt",
        confidence_threshold=0.60,
        confirmation_frames=4
    ):
        # Configuration
        self.arduino_port = arduino_port
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.confirmation_frames = confirmation_frames

        # Security state
        self.armed = False
        self.running = False
        self.person_detected = False
        self.confirmed_person = False

        self.people_count = 0
        self.confidence = 0.0
        self.latest_frame = None
        self.last_error = ""

        # Alert state
        self.confirmation_count = 0
        self.last_buzzer_command = None
        self.alert_active = False

        # Evidence and log paths
        self.evidence_dir = "evidence"
        self.log_file = "security_log.csv"

        os.makedirs(self.evidence_dir, exist_ok=True)

        # Create CSV file and header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(
                self.log_file,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "event",
                    "people_count",
                    "confidence",
                    "photo"
                ])

        # Thread synchronization
        self.lock = threading.RLock()

        # Hardware and AI objects
        self.arduino = None
        self.camera = None
        self.model = None
        self.worker = None

    def start(self):
        """Load YOLO and connect to Arduino and webcam."""

        if self.running:
            return

        try:
            self.last_error = ""

            print("Loading YOLO model...")
            self.model = YOLO(self.model_path)

            print("Connecting to Arduino:", self.arduino_port)

            self.arduino = serial.Serial(
                self.arduino_port,
                9600,
                timeout=0.1
            )

            # Allow the Arduino to reset after serial connection
            time.sleep(2)

            print("Opening webcam...")
            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Check camera access."
                )

            self.running = True

            self.worker = threading.Thread(
                target=self._camera_loop,
                daemon=True
            )
            self.worker.start()

            print("Security engine started.")
            print("Current mode: DISARMED")

        except Exception as error:
            self.last_error = str(error)
            print("Could not start security engine:", error)
            self.stop()

    def set_armed(self, armed):
        """Arm or disarm AI intrusion detection."""

        with self.lock:
            self.armed = bool(armed)
            self.confirmation_count = 0
            self.confirmed_person = False
            self.alert_active = False

        if not armed:
            self._send_buzzer(False)

        print(
            "Security mode:",
            "ARMED" if armed else "DISARMED"
        )

    def _send_buzzer(self, on):
        """Send 1 or 0 to the Arduino."""

        if self.arduino is None or not self.arduino.is_open:
            return

        command = "1" if on else "0"

        with self.lock:
            if command == self.last_buzzer_command:
                return

            try:
                self.arduino.write(command.encode("ascii"))
                self.last_buzzer_command = command

            except (serial.SerialException, OSError) as error:
                self.last_error = (
                    f"Arduino communication error: {error}"
                )

    def _save_evidence(self, frame, people_count, confidence):
        """Save an alert photo and append its details to the CSV."""

        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        filename = datetime.now().strftime(
            "intrusion_%Y%m%d_%H%M%S_%f.jpg"
        )

        photo_path = os.path.join(
            self.evidence_dir,
            filename
        )

        try:
            saved = cv2.imwrite(photo_path, frame)

            if not saved:
                self.last_error = "Could not save evidence photo."
                return

            with open(
                self.log_file,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    "PERSON_DETECTED",
                    people_count,
                    round(confidence, 3),
                    photo_path
                ])

            print("\nALERT: Person detected!")
            print("Time:", timestamp)
            print("People detected:", people_count)
            print("Confidence:", round(confidence, 3))
            print("Evidence photo:", photo_path)

        except (OSError, csv.Error) as error:
            self.last_error = (
                f"Evidence/log saving error: {error}"
            )

    def _camera_loop(self):
        """Read webcam frames, run YOLO, and handle alerts."""

        while self.running:
            success, frame = self.camera.read()

            if not success:
                self.last_error = "Unable to read webcam frame."
                time.sleep(0.1)
                continue

            try:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                # YOLO class 0 is the person class
                person_confidences = [
                    float(box.conf[0])
                    for box in results[0].boxes
                    if int(box.cls[0]) == 0
                ]

                detected = len(person_confidences) > 0

                max_confidence = max(
                    person_confidences,
                    default=0.0
                )

                annotated = results[0].plot()

                with self.lock:
                    # Require consecutive detections for confirmation
                    if detected:
                        self.confirmation_count += 1
                    else:
                        self.confirmation_count = 0

                    confirmed = (
                        detected
                        and self.confirmation_count
                        >= self.confirmation_frames
                    )

                    self.person_detected = detected
                    self.confirmed_person = confirmed
                    self.people_count = len(person_confidences)
                    self.confidence = max_confidence

                    armed_now = self.armed

                    # Save only when a new confirmed alert begins
                    save_alert = (
                        armed_now
                        and confirmed
                        and not self.alert_active
                    )

                    if armed_now and confirmed:
                        self.alert_active = True
                    else:
                        self.alert_active = False

                    # Display security status on the camera frame
                    if armed_now and confirmed:
                        status = "INTRUSION DETECTED"
                        color = (0, 0, 255)

                    elif armed_now:
                        status = "ARMED - MONITORING"
                        color = (0, 200, 255)

                    else:
                        status = "DISARMED"
                        color = (0, 255, 0)

                    cv2.putText(
                        annotated,
                        status,
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    self.latest_frame = annotated.copy()

                # Save evidence outside the state lock
                if save_alert:
                    self._save_evidence(
                        annotated,
                        len(person_confidences),
                        max_confidence
                    )

                # Activate buzzer only for confirmed detection while armed
                self._send_buzzer(
                    armed_now and confirmed
                )

            except Exception as error:
                self.last_error = (
                    f"Detection error: {error}"
                )
                time.sleep(0.1)

        # Attempt to turn off the AI-controlled buzzer on exit
        self._send_buzzer(False)

    def stop(self):
        """Stop the engine and release camera and serial resources."""

        self.running = False

        try:
            self._send_buzzer(False)
        except Exception:
            pass

        if (
            self.worker is not None
            and self.worker.is_alive()
            and threading.current_thread() != self.worker
        ):
            self.worker.join(timeout=3)

        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

        if self.arduino is not None:
            try:
                if self.arduino.is_open:
                    self.arduino.close()
            except Exception:
                pass
            self.arduino = None

        self.model = None
        self.worker = None

        print("Security engine stopped.")