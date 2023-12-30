# scRNA-seq Zero-Imputation Repository

This is the repository for the Zero Imputation library for the scRNA-seq research done at the NYU Open Networks and Big Data Lab.

## Setup instructions
1. Start a jupyter server either in terminal or browser
2. Install the relevant libraries in your virtual environment with the following: <br>`pip install -r requirements.txt`
3. Run the cells of the notebook `SERGIO/generate_data.ipynb`
4. Verify that the cells of the following notebooks run correctly: <br>
`SAUCIE/imputation_sergio_data.ipynb`
`scScope/imputation_SERGIO_data.ipynb`
`MAGIC/imputation_sergio_data.ipynb`
5. Run cells in `GENIE3/GENIE3_on_simulated_data.ipynb`


## Recent changelog
12/29/2023
- Added MAGIC imputation method to repository, and created notebook to generate appropriate data: `MAGIC/imputation_sergio_data.ipynb`
- Added test of MAGIC imputation within the GENIE3 benchmarking notebook `GENIE3/GENIE3_on_simulated_data.ipynb`
- SERGIO method incompatible with python versions > 3.9 (numpy.int depreciation), replaced `np.int()` instances with `int()` and `np.float()` with `float()` <br>
Impacted files: `sergio.py`
- Changed system path to be dynamic based on current user / program running scripts <br>
Impacted files: `SAUCIE/imputation_sergio_data.ipynb`, `scScope/imputation_SERGIO_data.ipynb`,
`GENIE3/GENIE3_on_simulated_data.ipynb`
- Fixed scope of scscope: changed from from `scscope ...` to from `scscope.scscope ...` <br>
Impacted files: `scScope/imputation_SERGIO_data.ipynb`