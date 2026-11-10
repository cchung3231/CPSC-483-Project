# Diabetes Prevention Model

## Project Overview

This project is developed for CPSC 483 – Introduction to Machine Learning at California State University, Fullerton.

The goal of this project is to explore diabetes-related health data and develop a machine learning model to help identify patterns and factors associated with diabetes risk.

The project includes data preprocessing, exploratory data analysis (EDA), visualization, model training, and evaluation.

---

## Getting Started

### Prerequisites

Before beginning, make sure you have the following installed:

- Python 3.12 (recommended)
- Visual Studio Code (VS Code)
- Git
- Python extension for VS Code
- Jupyter extension for VS Code

### 1. Clone the Repository

Open a terminal and clone the repository:

```bash
git clone <repository-url>
cd CPSC-483-Project
```

Replace `<repository-url>` with the GitHub repository URL.

### 2. Create a Virtual Environment

Using a virtual environment helps keep project dependencies separate from other Python projects.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
```

**Windows:**

```powershell
py -3.12 -m venv .venv
```

### 3. Activate the Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

Once activated, your terminal should display `(.venv)`.

### 4. Install Required Libraries

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses these libraries:

| Library | Purpose |
|---|---|
| pandas | Data analysis and manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualization |
| scikit-learn | Machine learning algorithms |
| jupyter | Interactive notebook execution |
| ipykernel | Jupyter kernel configuration |
| plotnine | Data visualization using Grammar of Graphics |



### 5. Configure VS Code

1. Open the project folder in VS Code.
2. Install the Python and Jupyter extensions.
3. Open the Command Palette (`Cmd + Shift + P` on macOS or `Ctrl + Shift + P` on Windows).
4. Search for `Python: Select Interpreter`.
5. Select the Python interpreter located in `.venv`.

### 6. Configure Jupyter Notebook

In the activated terminal, run:

```bash
python -m ipykernel install --user --name cpsc483 --display-name "CPSC 483 Python"
```

Then open your Jupyter notebook (`.ipynb`), click **Select Kernel** in the upper-right corner, and choose the project's `.venv` environment or the registered `CPSC 483 Python` kernel.

### 7. Verify Installation

Run the following code in a notebook cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from plotnine import *

print("All libraries imported successfully!")
```

If the message appears without errors, your environment is ready.

---

## Project Structure

```text
CPSC-483-Project/
├── Datasets/
│   └── pima-indians-diabetes-resampled.csv
├── .gitignore
├── README.md
├── requirements.txt
└── exploratory_analysis.ipynb
```

The `.venv` folder is created locally during setup and should not be committed to GitHub.

---

## Dataset

The project currently uses the `pima-indians-diabetes-resampled.csv` dataset, located in the `Datasets` directory.

To load the dataset:

```python
import pandas as pd

df = pd.read_csv("Datasets/pima-indians-diabetes-resampled.csv")

print(df.shape)
df.head()
```

---

## Troubleshooting

**ModuleNotFoundError**

Verify that the correct virtual environment is selected and dependencies have been installed:

```bash
python -m pip install -r requirements.txt
```

**Jupyter kernel not working**

Check that VS Code is using the correct interpreter and restart the notebook kernel.

**Dataset not found**

Confirm that you are working from the project root directory and that the dataset is located inside `Datasets/`.

---

## Git Collaboration

Before pushing changes, verify that `.gitignore` excludes local environments, Python cache files, and Jupyter checkpoints.

After installing or changing project dependencies, update the dependency list:

```bash
python -m pip freeze > requirements.txt
```

Review changes before committing.

---

## Authors

- David Ulloa

---

## Project Status

In Progress – Initial development environment configured. Data analysis and machine learning implementation are ongoing.
