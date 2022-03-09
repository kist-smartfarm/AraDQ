# Installation (2022.03 updated)
- 1.Download software (https://github.com/kist-smartfarm/AraDQ/archive/refs/heads/main.zip)
	- download as a zip (extracting a zip file)
		- ex. c:\leejaehoon\AraDQ
- 2.Install anaconda 
	- https://www.anaconda.com/products/individual
		- Python 3.7 (64-bit)
		- ref : https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/
- 2.Setting virtual environment using the Anaconda
	- ref : https://pythonforundergradengineers.com/new-virtual-environment-with-conda.html
- 3.Activate your virtual environment and install libararies
	- open Anaconda Prompt (Start Menu-Anaconda Prompt)
	- activate [name of virtual environment that you created]
		- ex) activate my_work_space
	- change the folder path
		- ex) cd c:\leejaehoon\AraDQ 
	- Please install build tools (for Windows user)
		- download and install it 
			- https://aka.ms/vs/16/release/vs_buildtools.exe
		- ref :https://www.scivision.co/python-windows-visual-c-14-required
	- Run 'python setup.py build_ext --inplace' on c:\leejaehoon\AraDQ (example)
		- ref : https://github.com/kist-smartfarm/AraDQ/tree/main/keras_retinanet
	- pip install -r requirements.txt on c:\leejaehoon\AraDQ (example)

- 4.Please download weight files
	- https://drive.google.com/file/d/1XTPQVzlqdVw5LM9_EYab3YiG9nqkFl3-/view?usp=sharing
		- extracting a zip file
		- put the 'ara_segmentation_model.h5' weight file and 'model.json' to the 'c:\leejaehoon\AraDQ\weights\ara_segmetnation' folder (example)
	- https://drive.google.com/file/d/11LdAtXLFsZAOYpdKNuc11B_2XaD7y66E/view?usp=sharing
		- extracting a zip file
		- put the 'color_card_model.h5' weight file to the 'c:\leejaehoon\AraDQ\weights\colorcard' folder (example)

- 5.Run local Flask server
	- python main.py
- 6.Open a browser (ex. Google Chrome)
	- Enter the 'localhost:20001'

# How to Use (AraDQ)

# Trouble shooting
- if you have a error for installing pydensecrf 
- please run this command
	- conda install -c conda-forge pydensecrf
	- ref : https://anaconda.org/conda-forge/pydensecrf
# Note
- This repository is tested using Python 3.6 (Windows 10).
- This repository is tested using Python 2.7 and 3.6 (Mac OSX 10.13.6 High Sierra).
