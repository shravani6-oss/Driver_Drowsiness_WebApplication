# facemeshdetect.py
import cv2
import mediapipe as mp
import numpy as np

class faceMeshDetection:
    def __init__(self, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Drawing utils
        self.mpdraw = mp.solutions.drawing_utils
        # Face mesh module
        self.face_mesh_module = mp.solutions.face_mesh
        # Initialize FaceMesh
        self.faceMesh = self.face_mesh_module.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def findfacemeshes(self, frame, draw_landmark=True):
        """
        Process frame and return landmarks.
        Returns:
            frame: Processed frame (with landmarks drawn if draw_landmark=True)
            faces: List of faces, each face is a list of (x, y) coordinates
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                face = []
                h, w, _ = frame.shape
                for lm in faceLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((cx, cy))
                faces.append(face)

                # Draw landmarks if requested
                if draw_landmark:
                    self.mpdraw.draw_landmarks(
                        frame,
                        faceLms,
                        self.face_mesh_module.FACEMESH_CONTOURS,
                        self.mpdraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

        return frame, faces
