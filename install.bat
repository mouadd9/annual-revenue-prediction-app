@echo off
echo Creating Python virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing project dependencies...
pip install -e .

echo Installation complete!
echo To activate the virtual environment, run: .venv\Scripts\activate.bat
echo To start Jupyter Notebook, run: jupyter notebook
pause 