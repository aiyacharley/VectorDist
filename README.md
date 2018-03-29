# VectorDist
Calculate the dist between two vector

## This module include TWO major section:
### 1. Normalize the vector
```
    1.1 MaxMIn
    1.2 Z_Score
    1.3 CountFrac
    1.4 sigmoid
```
### 2. Calculate the dist between two vectors
```
    2.1 Manhattan
    2.2 Euclidean
    2.3 Chebyshev
    2.4 Cosine
    2.5 Hamming
    2.6 Jaccard
    2.7 KLD(Kullback–Leibler divergence)
    2.8 JSD(Jensen-shannon divergence)
```
## usage: 
```
> from VectorDist import *
> Distance(Vector1,Vector2).Euclidean()
> Distance(Vector1,Vector2).JSD()
```
