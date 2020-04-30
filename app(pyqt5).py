# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QWidget, QGridLayout,QPushButton, QApplication)
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from utils import *
import joblib

# load api object
api = joblib.load('./model/api.pkl')

# window
class Window(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("PA廠計算機")
		self.setGeometry(50,50,350*2,350*2)
		self.UI()

	def UI(self):
		# answer area
		self.answer = QtWidgets.QTableView(self)
		self.answer.resize(800,800)
		
		# input area
		self.TextBox = QLineEdit(self)
		self.TextBox.setPlaceholderText('set_point')
		
		# button
		button = QPushButton("calculate",self)
		button.clicked.connect(self.getValues)
		
		# GridLayout
		self.gridLayout = QGridLayout()
		self.gridLayout.addWidget(self.TextBox,0,1)
		self.gridLayout.addWidget(button,1,1)
		self.gridLayout.addWidget(self.answer,2,1)
		self.setLayout(self.gridLayout)
		self.show()

	def getValues(self):
		set_point = float(self.TextBox.text())
		raw_advice = api.get_advice(set_point)
		advice = api.pretty_advice(raw_advice)
		output = api.get_critic_output(raw_advice)
		model = PandasModel(advice)
		self.answer.setModel(model)

# entry point
def main():
	App = QApplication(sys.argv)
	window = Window()
	sys.exit(App.exec_())

if __name__ == "__main__":
	main()

