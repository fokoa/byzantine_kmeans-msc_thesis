
# Fault resilient K-means byzantines

This is my Masters thesis and project done in the Department of Computer Science in the University of Yaoundé I. Supervised by [MELATAGIA YONTA Paulin](https://www.linkedin.com/in/paulin-melatagia-a27840115).


## Abstract

This thesis is based on the problem of distributed K-means in a Byzantine framework where we have $P$ machines which compute K centroids at each iteration. Among these P machines, a $\epsilon$-fraction ($\epsilon$ < 1/2) of them is Byzantine, and this Byzantine fraction has a tendency to compute K erroneous centroids, which, in most cases, will distort the distributed $K$-means algorithm. Therefore, to correct such Byzantine errors, we used a centroids aggregation rule; the FABA rule, which is an aggregation rule designed to aggregate the calculated gradient vectors in a distributed Byzantine environment. This thesis only conducts an experimental study on $K$-means coupled with FABA. Without providing any formal proof, we came to notice that the mixture of these two algorithms corrects the byzantine errors well up to a rate of 50% and allows us to obtain clusters quite close to the originals.

Keywords : `K-means`, `Distributed system`, `Byzantine`, `Gradient descent`.


## Use

After downloading this repository and installing the libraries contained in `requirements.txt`, go to the `code` folder and run one of the `main_*.py` files as follows:

`mpiexec -n 11 python3 main_*.py -b 5`

or

`mpiexec -n 11 python3 main_synthesized.py --byzantine 5`


* `11`: stand for total number of machines in the system (10 workers et 1 coordinator)
* `5`: stand for total number of byzantine workers (it must be at most half of the total number of machines)


## Bibtex

if you use this thesis, please cite:
```
@mastersthesis{fogang2021msc,
	author = {K. J. Fogang Fokoa},
	advisor = {Paulin Melatagia},
	title = {Fault resilient K-means byzantines},
	address = {Université de Yaoundé I, Yaoundé, Cameroun},
	year = {2021}
}
```
