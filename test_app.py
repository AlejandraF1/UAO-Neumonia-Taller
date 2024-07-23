
import pytest
import cv2
import numpy as np
from detector_neumonia import DetectorNeumonia

@pytest.fixture
def setup_detector():
    detector = DetectorNeumonia()
    return detector

def test_load_model(setup_detector):
    detector = setup_detector
    assert detector.model is not None

def test_preprocess(setup_detector):
    detector = setup_detector
    test_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    preprocessed_img = detector.preprocess(test_img)
    assert preprocessed_img.shape == (1, 512, 512, 1)

def test_grad_cam(setup_detector):
    detector = setup_detector
    test_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    detector.grad_cam(test_img)
    assert detector.heatmap is not None

def test_predict(setup_detector):
    detector = setup_detector
    test_img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    detector.predict(test_img)
    assert detector.label in ["bacteriana", "normal", "viral"]
    assert 0.0 <= detector.proba <= 100.0