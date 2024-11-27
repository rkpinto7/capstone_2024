# Capstone 2024

## Installation

1. Install Python 3 if you haven't done so already.
   - Go to [python.org](https://python.org) for downloads

2. Create a virtual environment:
   - Choose a location in which to create the virtual environment. 
   - Note: We call it "venv" and don't put it in the repo (it's in .gitignore)
   ```bash
   cd <location for your virtual environment folder>
   python3 -m venv venv
   ```

3. Activate your virtual environment using the appropriate command for your shell:
   - MAC zsh: `source venv/bin/activate`
   - MS cmd.exe: `venv\Scripts\activate.bat`
   - Linux csh: `source venv/bin/activate.csh`
   
   For other shells, see [Python venv documentation](https://docs.python.org/3/library/venv.html)

4. Install Django:
   - Upgrade pip to the most current version:
     ```bash
     python3 -m pip install --upgrade pip
     ```
   - Then install Django:
     ```bash
     python3 -m pip install django
     ```

5. Outside of your class repository, clone one of the class examples and try to run it:
   ```bash
   git clone https://github.com/jamesyiptk/capstone_2024.git
   cd capstone
   python3 manage.py migrate
   python3 manage.py runserver
   ```

6. Visit [http://localhost:8000](http://localhost:8000) and verify that the application is working.