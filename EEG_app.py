import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPainter
from mne.io import read_raw_edf
import mne
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def load_eeg_data(file_path):
    try:
        print(f"Loading EEG data from {file_path}...")
        raw_data = read_raw_edf(file_path, preload=True)
        raw_data.info['bads'] = []
        raw_data.pick_types(eeg=True)
        print("EEG data loaded successfully.")
        return raw_data
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        return None

def apply_filter(raw_data, low_freq, high_freq):
    try:
        print(f"Applying filter: Low {low_freq} Hz, High {high_freq} Hz...")
        raw_data.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
        print("Filter applied successfully.")
        return raw_data
    except Exception as e:
        print(f"Error applying filter: {e}")
        return None

def save_plot(fig):
    fig.savefig("filtered_eeg_plot.png")
    print("Plot saved as filtered_eeg_plot.png")

class LoadEEGThread(QThread):
    data_loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        raw_data = load_eeg_data(self.file_path)
        if raw_data is None:
            self.error.emit("Failed to load EEG data.")
        else:
            self.data_loaded.emit(raw_data)

class EEGApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Data Visualization App")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        filter_layout = QHBoxLayout()

        self.load_button = QPushButton("Load EEG Data")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.low_freq_slider = QSlider(Qt.Horizontal)
        self.low_freq_slider.setRange(0, 50)
        self.low_freq_slider.setValue(1)
        layout.addWidget(QLabel("Low Frequency"))
        layout.addWidget(self.low_freq_slider)

        self.high_freq_slider = QSlider(Qt.Horizontal)
        self.high_freq_slider.setRange(0, 50)
        self.high_freq_slider.setValue(50)
        layout.addWidget(QLabel("High Frequency"))
        layout.addWidget(self.high_freq_slider)

        self.plot_button = QPushButton("Plot EEG Data")
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        self.canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.load_thread = None
        self.raw_data = None

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open EEG File", "", "EDF Files (*.edf)")
        if file_path:
            print(f"Loading EEG data from {file_path}...")
            self.load_thread = LoadEEGThread(file_path)
            self.load_thread.data_loaded.connect(self.on_data_loaded)
            self.load_thread.error.connect(self.on_data_error)
            self.load_thread.start()

    def on_data_loaded(self, raw_data):
        self.raw_data = raw_data
        print("Data loaded successfully.")

    def on_data_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def plot_data(self):
        if self.raw_data is not None:
            low_freq = self.low_freq_slider.value()
            high_freq = self.high_freq_slider.value()
            filtered_data = apply_filter(self.raw_data, low_freq, high_freq)
            if filtered_data is not None:
                fig, ax = plt.subplots()
                filtered_data.plot_psd(fmin=low_freq, fmax=high_freq, ax=ax, show=False)
                self.canvas.figure = fig
                self.canvas.draw()
                save_plot(fig)
        else:
            QMessageBox.critical(self, "Error", "No EEG data loaded.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGApp()
    window.show()
    sys.exit(app.exec_())
