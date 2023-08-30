# A Stochastic Programming Approach to the Antibiotics Time Machine Problem

This repository contains the code for the paper A Stochastic Programming Approach to the Antibiotics Time Machine Problem

## Setup
1. Install dependencies:
 - python 3.7
 - numpy 1.19.5
 - pandas 1.3.4
 - networkx 2.5
 - gurobipy 10.0.0
 
Requirements can also be found in `requirements.txt`.

2. Clone the repository:`
git clone https://github.com/oguzmes/StochasticAntibiotic.git

3. Running Code and Getting Results
	Tool can be used from CLI after installing the necessary dependencies and cloning the repository. The information on how to use the tool is also defined within itself.
	```bash
	python ABR.py -h
	```
	```bash
	usage: StochasticAntibiotic [-h] [--dataset DATASET] [--n [N]] [--initialState [INITIALSTATE]]
	                            [--targetState [TARGETSTATE]] [--plotSolution]
	                            [--solutionMethod {DP,Multistage,Strong2stage,Weak2stage}]
	                            [--matrixSamplingSize MATRIXSAMPLINGSIZE] [--matrixType {epm,cpm}] [--timeLimit TIMELIMIT]
	
	Tool used to evaluate multiplication of matrices daha fazlasi, yazilabilir
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --dataset DATASET
	  --n [N]               step size (default: 3)
	  --initialState [INITIALSTATE]
	                        initial state selection (default: 1011)
	  --targetState [TARGETSTATE]
	                        target state selection (default: 0000)
	  --plotSolution        use if you want to plot solution (default: True)
	  --solutionMethod {DP,Multistage,Strong2stage,Weak2stage}
	                        solution method selection (default: DP)
	  --matrixSamplingSize MATRIXSAMPLINGSIZE
	                        matrix sampling size selection (default: 10000)
	  --matrixType {epm,cpm}
	                        matrix type selection (default: cpm)
	  --timeLimit TIMELIMIT
	                        time limit (seconds) for solvers (default: 3600)
	
	Developed by O. Mesum, Assoc. Prof. B. Kocuk
	```
	The following code will find best antibiotic treatment plan starting from genotype *1110* to genotype *0001* in *6* steps using *Strong2stage* method. Solver is time limit is set to *1000* seconds and matrix sampling size is *1000* for both evaluator and optimizer. If successful within time limit it will plot the solution aswell.
	```bash
	python ABR.py --dataset msx255_SuppData2017_GRME_ABR.xlsx --initialState 1110 --n 6 --targetState 0000 --solutionMethod Strong2stage
	```
	```bash
	Returning Matrix_useCase=optimization_type=cpm_s=10000 from existing file.
	Returning Matrix_useCase=evaluator_type=cpm_s=10000 from existing file.
	Saved N6_1110-0000_Strong2stage_cpm_solution.xlsx under ..\Data\Solutions.
	Saved N6_1110-0000_Strong2stage_cpm_plot.png under ..\Data\Solutions.	
	```

	Results of each instances can be found under  `..\Data\Solutions`.

## Data
The data used in this research can be found under repository named `msx255_SuppData2017_GRME_ABR.xlsx`.

## Known Issues 
The code has not been tested on Python versions other than 3.7.
The code may not work with versions of dependencies other than those specified above.
There aren't input checks for most of the cases.
Prior to solving instances the matrix sampling part is taking very long. So for this if a new matrix samples are generated the matrices are saved under `..\Data\MatrixFiles`. Note in the previous example we haven't run an instance with matrix sampling size of 100 before. So to avoid generating matrix with 100 samples again, it will save the created matrices for future use.


  
## License 
The code in  this repository is released under the [MIT License](https://opensource.org/licenses/MIT).


## Citing
If you use this code or the results from our paper in your own work, please cite our paper:`
```sql
 @article{stoch-antibiotic,
  title={Stochastic Antibiotic},
  author={},
  journal={},
  volume={},
  pages={},
  year={2023}
}
```
