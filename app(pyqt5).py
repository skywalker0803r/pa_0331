import sys
from PyQt5.QtWidgets import *

class Window(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("PA廠計算機")
		self.setGeometry(50,50,350,350)
		self.UI()

	def UI(self):
		#================================
		self.answer = QLabel("",self)
		self.answer.resize(200,100)
		self.answer.setFrameStyle(QFrame.Panel | QFrame.Sunken)
		self.answer.move(80,200)
		#================================
		self.nameTextBox = QLineEdit(self)
		self.nameTextBox.setPlaceholderText('set_point')
		self.nameTextBox.move(120,50)
		#==============================
		self.passTextBox = QLineEdit(self)
		self.passTextBox.setPlaceholderText('feed')
		self.passTextBox.move(120,80)
		#==============================
		button = QPushButton("calculate",self)
		button.move(120,110)
		button.clicked.connect(self.getValues)
		#==============================
		self.show()

	def getValues(self):
		name = self.nameTextBox.text()
		password = self.passTextBox.text()
		result = 'control factor1:'+str(float(name) + float(password))
		result = result +'\n'+ result
		self.answer.setText(result)

# exec
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec_())

