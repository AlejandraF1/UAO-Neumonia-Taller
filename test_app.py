import pytest
import numpy as np
from detector_neumonia import DetectorNeumonia, read_jpg_file, preprocess

@pytest.fixture
def detector():
    return DetectorNeumonia()

def test_load_img_file(detector, mocker, tmp_path):
    img_file = tmp_path / 'image.jpg'
    img_file.touch()  # create a temporary image file
    mocker.patch('filedialog.askopenfilename', return_value=str(img_file))
    detector.load_img_file()
    assert detector.array is not None

def test_run_model(detector, mocker):
    mocker.patch('detector_neumonia.predict', return_value=('label', 0.5, 'heatmap'))
    detector.array = np.array([1, 2, 3])
    detector.run_model()
    assert detector.label == 'label'
    assert detector.proba == 0.5
    detector_neumonia.predict.assert_called_once_with(detector.array)  # verify predict was called with detector.array

def test_save_results_csv(detector, mocker, tmp_path):
    csv_file = tmp_path / 'results.csv'
    mocker.patch('csv.writer', return_value=mocker.Mock())
    detector.text1.insert(0, 'patient_id')
    detector.label = 'label'
    detector.proba = 0.5
    detector.save_results_csv(str(csv_file))
    assert detector.text1.get() == 'patient_id'
    assert detector.label == 'label'
    assert detector.proba == 0.5
    # verify CSV file was written correctly
    with open(csv_file, 'r') as f:
        assert f.read() == 'patient_id,label,0.5\n'