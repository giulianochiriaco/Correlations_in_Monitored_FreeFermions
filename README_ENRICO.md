# Free_fermios_functions.py Documentation

## Overview
The `Free_fermios_functions.py` script provides a collection of functions related to the study and simulation of free fermionic systems. The functions included in this script are designed to help users perform various computational tasks such as diagonalization of Hamiltonians, computation of correlation functions, and analysis of fermionic observables.

## Features
- Diagonalization of matrices related to fermionic Hamiltonians.
- Computation of various fermionic observables.
- Utilities for handling fermion creation and annihilation operators.
- Support functions for numerical and analytical analysis in free fermion models.

## Installation
1. Ensure you have Python 3 installed on your system.
2. Install required libraries using pip (if any are specified in the script):
	```
	pip install numpy scipy matplotlib
	```
3. Place the `Free_fermios_functions.py` script in your working directory.

## Usage
Import the functions provided in the script into your project:

## How does `Transient_Cr_avg` work
1. Creates a half-filled chain for both chains (`v1` and `v2`) and shuffles them, and concatenates them in the diagonal matrix `D0`
1. Defines `NT` (useful for `t_step` different from 1) that represents the number of measurements **saved** (the performed measures are always `Nmax`)
1. Defines the indexes for the measurements: 
	1. `r_1` indexes from `0` to `L/2+1` (taking in account PBCs)
	1. `r_12` indexes from `0` to `L` (taking in account PBCs yet considering both chains)
	1. `j_values` **column vector** of indexes to set starting site for the correlation measurement
	1. `jp_1` (and similarly `jp_12` for the cross correlations) is a matrix that gives the indexes of the elements correlated:
1. if measures are to be collected defines `D1` as the mod squared of the elements of `D0` and splits it into the three blocks of interes (`11` for chain 1 self correlation, `12` for cross correlation, `22` for chain 2 self correlation) and evaluates the mean value of the correlation at $r=0, 1, \dots, \frac{L}{2}+1$

### Example 

For the sake of clarity, let us consider the simple case of `L=8`. In this case we look at a `D11` matrix whose elements are: 

	D11 = [[ 0,  1,  2,  3,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29, 30, 31],
       [32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47],
       [48, 49, 50, 51, 52, 53, 54, 55],
       [56, 57, 58, 59, 60, 61, 62, 63]]

and we obtain that:

	j_values = [[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7]]

so that, when taking, e.g. `j_values + r_1`:

	jp_1 = [[0, 1, 2, 3, 4],
       [1, 2, 3, 4, 5],
       [2, 3, 4, 5, 6],
       [3, 4, 5, 6, 7],
       [4, 5, 6, 7, 0],
       [5, 6, 7, 0, 1],
       [6, 7, 0, 1, 2],
       [7, 0, 1, 2, 3]]

and the mean is evaluated over the columns (`axis=0`) of the following:

	D11[j_values,jp_1] =
	  [[ 0,  1,  2,  3,  4],
       [ 9, 10, 11, 12, 13],
       [18, 19, 20, 21, 22],
       [27, 28, 29, 30, 31],
       [36, 37, 38, 39, 32],
       [45, 46, 47, 40, 41],
       [54, 55, 48, 49, 50],
       [63, 56, 57, 58, 59]]

## Saved data
At each iteration the correlations obtained as described in the previous section are saved to a file in the following format:

	[t_vec]
	[[Correlation as a function of r at each t_step:
		C(r=0, t=0)	C(r=1, t=0) 	... C(r=L/2+1, t=0)
		C(r=0, t=1)	C(r=1, t=1) 	... C(r=L/2+1, t=1)
		...
		C(r=0, t=t_final)	C(r=1, t=t_final) 	... C(r=L/2+1, t=t_final)
		]]

where `t_final = Nmax/t_step`, and `t_vec` specifies the time steps to which each row corresponds. In each file this structure is replicated for a number of times equal to `NRseries` that is the total number of trajectories generated.