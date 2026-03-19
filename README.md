# Hello and welcome to SpotMini's software

This github repository was repurposed for our capstone project from the original author, Maurice Rahme, and the SpotMini community. 

To get started, please download this repository first

Access the spot_bullet folder -> src and please find the requirements.txt

We will be running all these files on our terminal so access your terminal and navigate (cd ~) to the spot_bullet folder then into src. I have directions right below for 3 types of computers. Please follow the directions for your computer and download all the libraries in the requirements.txt.

From spot_bullet, please find the src folder, and please find spot_tester.py

Path = ________/spot_mini_mini/spot_bullet/src







## FOR WINDOWS USERS:

### 1. Create environment
python -m venv spotmini-env

### 2. Activate
spotmini-env\Scripts\activate

### 3. Upgrade pip
pip install --upgrade pip setuptools wheel

### 4. Install dependencies (Please get rid of # for pybullet and opencv)
pip install -r requirements.txt

### 5. Run
python src\spot_tester.py




## FOR MAC(INTEL) USERS:

### 1. Create environment
python3 -m venv spotmini-env

### 2. Activate
source spotmini-env/bin/activate

### 3. Upgrade pip (VERY IMPORTANT)
pip install --upgrade pip setuptools wheel

### 4. Install dependencies
pip install -r requirements.txt

### 5. Run
python src/spot_tester.py





## FOR MAC(APPLE Silicon) USERS:

### 1. Create environment
conda create -n spotmini python=3.10 -y

### 2. Activate
conda activate spotmini

### 3. Install core dependencies (prevents Mac issues)
conda install -c conda-forge numpy scipy matplotlib opencv pybullet -y

### 4. Upgrade pip
pip install --upgrade pip wheel

### 5. Pin setuptools because pkg_resources was removed in setuptools 82+
pip install "setuptools<82"

### 6. Install secondary dependencies
pip install gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.3.2 filterpy==1.4.5

### 5. Install remaining dependencies
pip install -r requirements.txt

### 6. Run
python spot_tester.py








## Citing Spot Mini Mini
```
@software{spotminimini2020github,
  author = {Maurice Rahme and Ian Abraham and Matthew Elwin and Todd Murphey},
  title = {SpotMiniMini: Pybullet Gym Environment for Gait Modulation with Bezier Curves},
  url = {https://github.com/moribots/spot_mini_mini},
  version = {2.1.0},
  year = {2020},
}
```

## Credits

* Original Spot Design and CAD files: [Spot Micro AI Community](https://spotmicroai.readthedocs.io/en/latest/)

* Collaborator on `OpenQuadruped` design, including mechanical parts, custom PCB, and Teensy interface: [Adham Elarabawy](https://github.com/adham-elarabawy/OpenQuadruped)

* OpenAI Gym and Heightfield Interface: [Minitaur Environment](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py)

* Deprecated URDF for earlier development: [Rex Gym](https://github.com/nicrusso7/rex-gym)

Note: development for this project was haulted in November 2020 to respect my NDA with my employer.
