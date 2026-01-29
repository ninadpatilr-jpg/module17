# Practical Application III — Comparing Classifiers with Bank Marketing (prompt_III.ipynb)

This repository contains a Jupyter Notebook, `prompt_III.ipynb`, that explores and models the UCI Bank Marketing dataset. The notebook walks through data loading, exploratory data analysis (EDA), feature preparation, baseline classification models, and hyperparameter tuning using GridSearchCV.

## Repository Structure
- `prompt_III.ipynb` — Main notebook covering EDA, feature engineering, model training, and tuning
- `data/` — Input CSV files required by the notebook (e.g., `bank-additional-full.csv`)

## Environment Setup
Install the required Python packages (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyterlab nbconvert
```

Alternatively, using `conda`:

```bash
conda create -n bankenv python=3.11 -y
conda activate bankenv
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyterlab nbconvert
```

## Running the Notebook
1. Activate your environment (virtualenv or conda environment above).
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook prompt_III.ipynb
# or
jupyter lab prompt_III.ipynb
```

3. Recommended execution flow:
   - Run the imports/setup cell.
   - Run the data loading cell: `df = pd.read_csv('data/bank-additional-full.csv', sep=';')`.
   - Execute EDA and plotting cells to inspect the data.
   - Run model training and hyperparameter search cells when ready. Note: the GridSearch cell can be time-consuming.

## Quick verification (headless)
To execute the notebook non-interactively and save outputs in-place:

```bash
jupyter nbconvert --to notebook --execute prompt_III.ipynb --ExecutePreprocessor.timeout=1200 --inplace
```

Increase `--ExecutePreprocessor.timeout` if long-running cells (e.g., GridSearch) need more time.

## Tips
- To speed verification of the GridSearch cell, reduce parameter grid sizes or `cv` folds.
- If you run into environment/kernal issues with `nbconvert`, open the notebook interactively and ensure the kernel matches the Python environment where dependencies are installed.

## Findings
- **Dataset / target handling:** The notebook uses the UCI Bank Marketing dataset. Because the target variable may appear under different names across dataset versions (e.g., `deposit`, `Deposit`, or `y`), the notebook includes logic to automatically detect the correct column and apply a safe fallback to prevent runtime errors..
- **EDA highlights:** Visualizations summarize customer attributes such as job type and marital status. Certain plots use age as a consistent counting column to ensure robustness across dataset variants. Plotly visualizations are standardized to avoid column mismatch issues.
- **Model / GridSearch summary:** GridSearchCV is applied to four classifiers—Logistic Regression, Decision Tree, KNN, and SVC. Representative best cross-validation accuracies observed in one execution were:
   - features used - **`age`, `pdays`, `njob`, `campaign`, `neducation`**
   - Logistic Regression: **0.8971**, best params: `{'clf__C': 0.1, 'clf__solver': 'lbfgs'}`
   - Decision Tree: **0.8971**, best params: `{'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}`
   - KNN: **0.8947**, best params: `{'clf__metric': 'minkowski', 'clf__n_neighbors': 11, 'clf__weights': 'uniform'}`
   - SVM: **0.8973**, best params: `{'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}`
   - Best overall: **SVM  0.8973373348115092**
- **Operational note:** GridSearch in the notebook is configured to run with `n_jobs=1` to avoid platform-specific `ChildProcessError` issues; if you have a stable multiprocessing environment, you can increase `n_jobs` for speed.

## Next suggestions
- Execute the notebook end-to-end to validate results and ensure there are no remaining runtime or logic issues.
- Enhance model evaluation by persisting the best-performing model and adding diagnostic metrics such as confusion matrices, ROC curves, and AUC scores.
- Incorporate customer account balance as a feature, if available. Account balance is likely to be a strong predictor of term-deposit subscription, under the assumption that customers with higher balances have greater propensity to invest in long-term deposits.
- Explore Deep Learning approaches, such as Artificial Neural Networks (ANNs), to assess whether non-linear models can further improve predictive performance compared to traditional classifiers.


Generated on: `2026-01-27`
Updated on: `2026-01-28`
Author: `ninadpatilr-jpg`
Notebook path: `prompt_III.ipynb`
