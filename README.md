# Hello and welcome to SpotMini's software

This github repository was repurposed for our capstone project from the original author, Maurice Rahme, and the SpotMini community.

To get started, please download this repository first.

Access the spot_bullet folder -> src and please find the requirements.txt.

We will be running all these files on our terminal so access your terminal and navigate to the spot_bullet folder then into src. I have directions right below for 3 types of computers. Please follow the directions for your computer and download all the libraries in the requirements.txt.

From spot_bullet, please find the src folder, and please find spot_tester.py.

Path = ________/spot_mini_mini



## FOR WINDOWS USERS:

For Windows, I have had the most success using Miniconda or Anaconda instead of a plain `venv`.

https://www.anaconda.com/docs/getting-started/miniconda/install/overview

### 1. Install Python 3.10
You can download the latest, but while running the next bit of code we will be creating an environment using python 3.10
Make sure Python 3.10 is available through your Conda installation.

https://www.python.org/downloads/

### 2. Open the correct terminal
Open `Anaconda Prompt` or `Miniconda Prompt`.

### 3. Go to the project folder
```bat
cd C:\Users\YOUR_USERNAME\Downloads\spot_mini_mini
```

### 4. Create the environment
```bat
conda create -n spotmini python=3.10 -y
```

### 5. Activate the environment
```bat
conda activate spotmini
```

### 6. Install the core dependencies with conda-forge
These packages were more reliable through Conda on Windows than through pip.

```bat
conda install -c conda-forge numpy scipy matplotlib opencv pybullet -y
```

### 7. Upgrade pip tools
```bat
python -m pip install --upgrade pip wheel "setuptools<82"
```

### 8. Install the remaining Python dependencies
```bat
python -m pip install gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.3.2 filterpy==1.4.5
```

### 8.5. Install the remaining Python dependencies
```bat
python -m pip install spot_bullet/src/requirements.txt
```

### 9. Verify PyBullet is working
```bat
python -c "import pybullet; print('pybullet works')"
```

### 10. Start the tester
Run this from the main project folder:

```bat
python spot_bullet\src\spot_tester.py
```

#### 11. Compatibility note
`spot_tester.py` now includes a fallback for older copies of `spotmicro/util/gui.py`.

If `IndividualLegGUI` is not available in your local `gui.py`, the tester will automatically fall back to the standard `GUI` class instead of stopping with an import error.




## FOR MAC(APPLE Silicon) USERS:

For Apple Silicon, we also had the most success using Conda instead of a plain `venv`.

### 1. Install Python 3.10
Make sure Python 3.10 is available through your Conda installation.

### 2. Open the correct terminal
Open your normal macOS Terminal with Conda initialized.

### 3. Go to the project folder
```bash
cd /path/to/spot_mini_mini
```

### 4. Create the environment
```bash
conda create -n spotmini python=3.10 -y
```

### 5. Activate the environment
```bash
conda activate spotmini
```

### 6. Install the core dependencies with conda-forge
These packages were more reliable through Conda than through pip on Apple Silicon.

```bash
conda install -c conda-forge numpy scipy matplotlib opencv pybullet -y
```

### 7. Upgrade pip tools
```bash
python -m pip install --upgrade pip wheel "setuptools<82"
```

### 8. Install the secondary Python dependencies
```bash
python -m pip install gym==0.26.2 gymnasium==0.29.1 stable-baselines3==2.3.2 filterpy==1.4.5
```

### 8.5. Install final dependencies
```bash
python -m pip install -r spot_bullet/src/requirements.txt
```

### 9. Start the tester
Run this from the main project folder:

```bash
python spot_bullet/src/spot_tester.py
```




## FOR MAC(INTEL) USERS:

### 1. Create environment
python3 -m venv spotmini-env

### 2. Activate
source spotmini-env/bin/activate

### 3. Upgrade pip
pip install --upgrade pip setuptools wheel

### 4. Install dependencies
pip install -r requirements.txt

### 5. Run
python src/spot_tester.py




# Manual PyBullet App Setup

This is the new app-style manual controller. It opens a local app window and lets you start a native PyBullet GUI.

This is different from the older dashboard because manual control has its own file.

Files used:

display_manual.py = the manual control app

launch_manual_app.py = starts the manual control app

manual_pybullet_worker.py = starts and controls the PyBullet simulation

Path = ________/spot_mini_mini/spot_bullet/Sim Display

Before running the manual app, install the app dependencies in the same environment:

### python -m pip install streamlit pywebview



## Run on Windows:

### python "spot_bullet\Sim Display\launch_manual_app.py"


## Run on Mac:

### python "spot_bullet/Sim Display/launch_manual_app.py"



## What to click in the app:

Click Manual Mode if you want to control the dog with the on screen buttons.

Click Test PPO Model if you want to test one of the trained reinforcement learning models.

For Manual Mode, choose the gait button, choose the terrain button, and then click Launch Manual Mode.

For Test PPO Model, choose the trained run from the dropdown, choose best or final, and then launch the worker.

If the worker is still using an older version, click Force Restart Manual Worker.

Manual trot currently uses fixed tuning values while we are still improving the gait. The current values are forward step 0.050, backward step 0.040, turn assist 0.010, turn rate 0.90, strafe step 0.030, strafe angle 0.80, step velocity 0.54, and curve height 0.040.




# Main Dashboard Setup

The original dashboard is still separate from the new manual app.

Path = ________/spot_mini_mini/spot_bullet/Sim Display



## Run on Windows:

### python "spot_bullet\Sim Display\launch_dashboard.py"

If pywebview is installed correctly, the dashboard opens in a local desktop window. If pywebview is missing, the normal dashboard can open in a browser.


## Run on Mac:

### python "spot_bullet/Sim Display/launch_dashboard.py"



# Important note for the newer commands

The updated Windows and Apple Silicon `spot_tester.py` commands above are run from the main project folder:

Path = ________/spot_mini_mini

If you prefer to run `spot_tester.py` from inside `spot_bullet/src`, use:

```bash
python spot_tester.py
```

The newer dashboard, manual PyBullet app, training, playback, and comparison commands below should be run from the main project folder:

Path = ________/spot_mini_mini

If you are in the main project folder and need to install the requirements file, use:

```bash
python -m pip install -r spot_bullet/src/requirements.txt
```

If you are already inside spot_bullet/src, use:

```bash
pip install -r requirements.txt
```





# Training a PPO model

Training creates a new reinforcement learning run inside:

spot_bullet/training runs

## Quick smoke test:

python3 spot_bullet/src/spot_train_ml.py --timesteps 2048 --n-envs 1 --skip-env-check --preview-freq 0 --run-name yaw_smoke_test --enable-imu-yaw

## Rough terrain training preset:

python3 spot_bullet/src/spot_train_ml.py --training-preset quick_rough --run-name rough_longwalk_yaw_transfer_v1 --enable-imu-yaw

The training code now looks for runs in both folder names:

spot_bullet/training runs

spot_bullet/training_runs

This helps avoid problems when older runs used the underscore folder and newer runs use the folder with a space.





# Playing or visualizing a trained PPO model

Use this when you want to test a trained model.

## Run the newest available run without visualization:

python3 spot_bullet/src/spot_play_ml.py --episodes 1

## Run a specific model with the native PyBullet GUI:

python3 spot_bullet/src/spot_play_ml.py --run-dir "spot_bullet/training runs/yaw_smoke_test" --model final --episodes 1 --render --bullet-gui

Use --model best to load the best saved model.

Use --model final to load the final saved model.

If you do not see the native PyBullet window, make sure you used both:

--render --bullet-gui





# Comparing trained PPO models

Use this when you want to compare two or more trained models without opening the GUI.

python3 spot_bullet/src/spot_compare_ml.py "spot_bullet/training runs/no_yaw_test" "spot_bullet/training runs/with_yaw_test" --model final --episodes 5 --sort-by reward

The comparison report shows reward, speed, distance, height error, falls, and timeouts.





# Crash testing report

For the assignment where we deliberately try to crash the platform, use:

spot_bullet/CRASH_TEST_REPORT.md

This file has the attack commands, what screenshots to take, what the expected outcome is, and what was implemented to prevent the same issue from crashing the platform again.





# Windows fixes that were added for the dashboard and manual app

The dashboard and manual app now check if the computer is Windows before stopping or starting worker processes.

On Windows, the app uses normal process terminate and kill behavior.

On Mac, the app can still use process groups for cleaner shutdown.

This was added in:

spot_bullet/Sim Display/launch_dashboard.py

spot_bullet/Sim Display/launch_manual_app.py

spot_bullet/Sim Display/display.py

spot_bullet/Sim Display/display_manual.py

This helps the dashboard and manual PyBullet app close workers more safely on Windows.





# Current important files

spot_bullet/Sim Display/display.py = main dashboard UI

spot_bullet/Sim Display/launch_dashboard.py = starts the main dashboard

spot_bullet/Sim Display/display_manual.py = manual control app UI

spot_bullet/Sim Display/launch_manual_app.py = starts the manual control app

spot_bullet/Sim Display/manual_pybullet_worker.py = PyBullet worker used by manual mode and PPO model testing

spot_bullet/src/spot_train_ml.py = PPO training file

spot_bullet/src/spot_play_ml.py = PPO playback file

spot_bullet/src/spot_compare_ml.py = PPO model comparison file

spot_bullet/src/spot_ml.py = reinforcement learning environment file

spotmicro/GaitGenerator/Bezier.py = Bezier gait generator file





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
