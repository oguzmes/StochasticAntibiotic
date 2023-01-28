# Research Title

This repository contains the code for the paper ...

## Setup
1. Install dependencies:
 - python 3.7
 - numpy 1.19.5
 - pandas 1.3.4
 - networkx 2.5
 - gurobipy 10.0.0
 
Requirements can also be found in `requirements.txt`.

2. Clone the repository:`
git clone https://github.com/oguzmes/StochastiAntibiotic.git

3. Running Code and Getting Results
	Tool can be used from CLI after installing the necessary dependencies and cloning the repository. The information on how to use the tool is also defined within itself.
	```sh
	python ABR.py -h
	```
	```sh
	usage: StochasticAntibiotic [-h] [-n [N]] [-is [INITIALSTATE]]
                            [-ts [TARGETSTATE]] [-ps]
                            [-sm {DP, Multistage, Strong2stage, Weak2stage}]
                            [-mss MATRIXSAMPLINGSIZE] [-mt {epm,cpm}]
                            [-tl TIMELIMIT]
                            dataset

	Tool used to evaluate multiplication of matrices daha fazlasi, yazilabilir

	positional arguments:
	  dataset

	optional arguments:
	  -h, --help            show this help message and exit
	  -n [N], --n_stepsize [N]
	                        step size (default: 4)
	  -is [INITIALSTATE], --initialState [INITIALSTATE]
	                        initial state selection (default: 1111)
	  -ts [TARGETSTATE], --targetState [TARGETSTATE]
	                        target state selection (default: 0000)
	  -ps, --plotSolution   use if you want to plot solution (default: False)
	  -sm {DP, Multistage, Strong2stage, Weak2stage}, --solutionMethod {DP, Multistage, Strong2stage, Weak2stage}
	                        solution method selection (default: DP)
	  -mss MATRIXSAMPLINGSIZE, --matrixSamplingSize MATRIXSAMPLINGSIZE
	                        matrix sampling size selection (default: 10000)
	  -mt {epm,cpm}, --matrixType {epm,cpm}
	                        matrix type selection (default: cpm)
	  -tl TIMELIMIT, --timeLimit TIMELIMIT
	                        time limit (seconds) for solvers (default: 3600)

	Developed by O. Mesum, Assoc. Prof. B. Kocuk
	```
	The following code will find best antibiotic treatment plan starting from genotype *1110* to genotype *0001* in *6* steps using *Strong2stage* method. Solver is time limit is set to *1000* seconds and matrix sampling size is *1000* for both evaluator and optimizer. If successful within time limit it will plot the solution aswell.
	```sh
	python ABR.py data.xlsx -is 1110 -n 6 -ts 0001 -sm Strong2stage -mss 100 -tl 1000 -ps
	```
	```sh
	Matrix_useCase=optimization_type=cpm_s=100 does not exist, generating from scratch.
	Matrix_useCase=evaluator_type=cpm_s=100 does not exist, generating from scratch.
	Saving results N6_1110-0001_Strong2stage_cpm_solution.xlsx
	Saving plot N6_1110-0001_Strong2stage_cpm_plot.png	
	```

	
	Results of each instances can be found under  `..\Data\Solutions`.

## Data
The data used in this research can be found under repository named `data.xlsx`.

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