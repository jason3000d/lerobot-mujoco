# Robot Simulation Setup Guide

This document provides instructions for installing and running the robot simulation package.

## Prerequisites

1. Download the ZIP file.
2. Extract the contents of the ZIP file to your desired location.

---

# Windows Installation

1. Navigate to the location where the files were extracted.
2. Open a terminal using one of the following methods:
   - Right-click inside the extracted folder and select Open in Terminal.
   - Open PowerShell from the Start menu and navigate to the extracted folder.

3. Run the following command:

powershell -ExecutionPolicy Bypass -File setup.ps1

---

# Linux Installation

1. Navigate to the location where the files were extracted.
2. Open a terminal in the extracted folder.
3. Make the setup script executable: 
	chmod +x ./setup.sh

4. Run the setup script:
	./setup.sh

---
# Run Simulation:

Linux:

   uv run python robo-code/sim.py explore

Windows:

   uv run python robo-code/sim.py explore

Mac:

   uv run mjpython robo-code/sim.py explore

---



# Project Structure

```
project-root/
├── robot-code/
├── robot-assets/
├── setup.ps1
├── setup.sh
└── README.md
```

## Folders

### robot-code

This folder contains the code required to run the simulation.

A sample program is included that:

- Runs the simulator for 10 minutes.
- Prints the robot's current joint angles to the terminal.

### robot-assets

This folder contains the model files required by the simulator to load the robot model correctly.

Note: Do not modify any files in this folder, as doing so may prevent the simulator from functioning properly.

---

# Support

If you encounter any issues during installation or execution, verify that:

- The ZIP file was extracted completely.
- You are executing the setup script from the correct directory.
- The required permissions have been granted for script execution.

Windows:
- Error: The term 'uv' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the
spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
- Solution: If you enounter this error close the current terminal and open a new one.

Mac:
- Error: zsh: command not found: uv
- Solution: Run the following command in the terminal before running the program: export PATH="$HOME/.local/bin:$PATH"

Linux:
- Error: uv command not found
- Solution: Close the temrinal and reopen it.