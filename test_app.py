import pytest
import numpy as np
from app import App, read_jpg_file, preprocess
import sys
sys.path.append('../')

@pytest.fixture
def app():
    return App()

def test_load_img_file(app, mocker, tmp_path):
    img_file = tmp_path / 'image.jpg'
    img_file.touch()  # create a temporary image file
    mocker.patch('filedialog.askopenfilename', return_value=str(img_file))
    app.load_img_file()
    assert app.array is not None

def test_run_model(app, mocker):
    mocker.patch('predict', return_value=('label', 0.5, 'heatmap'))
    app.array = np.array([1, 2, 3])
    app.run_model()
    assert app.label == 'label'
    assert app.proba == 0.5
    predict.assert_called_once_with(app.array)  # verify predict was called with app.array

def test_save_results_csv(app, mocker, tmp_path):
    csv_file = tmp_path / 'results.csv'
    mocker.patch('csv.writer', return_value=mocker.Mock())
    app.text1.insert(0, 'patient_id')
    app.label = 'label'
    app.proba = 0.5
    app.save_results_csv(str(csv_file))
    assert app.text1.get() == 'patient_id'
    assert app.label == 'label'
    assert app.proba == 0.5
    # verify CSV file was written correctly
    with open(csv_file, 'r') as f:
        assert f.read() == 'patient_id,label,0.5\n'