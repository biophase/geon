from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton,QVBoxLayout, QSlider
import sys



app = QApplication([])

window = QWidget()
window.setWindowTitle("test app")
# window.setGeometry(50,50,1000,200)
layout = QVBoxLayout(window)

label = QLabel("<h1> This is a message </h1>")
button = QPushButton("Click here.")
slider = QSlider()
slider.setMinimum(0)
slider.setMaximum(100)

button.clicked.connect(lambda: label.setText("Button got clicked."))


layout.addWidget(label)
layout.addWidget(button)
layout.addWidget(slider)

window.show()


# event loop
app.exec()
