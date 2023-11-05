# Introduction

A _Normalizing Flow_ is a transformation of a complicated probability distribution into a given simple distribution, often a ``D``-dimensional standard Normal distribution. 
    
Normalizing Flows consist of bijective functions that act on samples drawn from the complex distribution so that the transformed samples follow the simple distribution. 
Using the inverse of such a Normalizing Flow, one may also transform samples drawn from a simple distribution so that the transformed samples follow a  complex target distribution.

Real-world data often follow challenging probability distributions, making statistical methods difficult or impossible to apply. Normalizing Flows can alleviate this difficulty by shifting the focus from the complex distribution itself to finding a transformation from it to a simple distribution. The choice of this simple distribution is arbitrary. The ``D``-dimensional _Normal_ distribution is the prevalent option, hence the name "Normalizing Flows". 

Tabak and Vanden-Eijnden [^1] first defined the framework of the method in 2010. It was popularized in the following years up to 2015 when several groups in machine learning explored Normalizing Flows in the context of clustering and classification [^2], density estimation [^3] [^4] [^5], and variational inference [^6]. 

In the context of machine learning, Normalizing Flows are often introduced as generative models as one of their main practical applications is the _sampling_ of a target distribution. Another common use case is _density evaluation_ [^7].


## Basics 

A Normalizing Flow is a method that transforms one probability distribution into another via a series of invertible and differentiable mappings. 
        
A typical application of Normalizing Flows is as follows: Consider a random variable ``\bm{X} \sim p_X``, where ``p_X`` is some intractable or unknown distribution. By sampling or observations, a set of samples from ``p_X`` is obtained, and a known and simple distribution ``p_Y`` (usually a standard Gaussian) is chosen as the outcome of the Flow. Then an invertible and differentiable function ``\bm{f}`` is generated so that:
```math
    \bm{f}(\bm{X}) = \bm{Y} \sim p_Y.
```
This function ``\bm{f}`` is the Normalizing Flow. It achieves the "normalization" of the target distribution ``p_X`` by _squeezing_ and _stretching_ it into the shape of ``p_Y``, see figure 1. 

![Alt text](/assets/2spaces.png "Working principle of a Normalizing Flow")

This is very useful for statistical analysis, as performing tasks such as inference on simple distributions is trivial. 
Given such a Normalizing Flow, it is, for example, possible to do easy and precise _density estimation_ of ``p_X`` and to generate new samples ``\bm{x} \sim p_X`` [^7], tasks that would otherwise be challenging. Of course, obtaining such a Flow is often a challenge in itself.

A Normalizing Flow's _invertibility_ and _differentiability_ are the essential properties that make the abovementioned applications possible. To efficiently apply Normalizing Flows in integration tasks they also need to be easily computable.


### Density Evaluation
Given some data ``\mathcal{D} = \{\bm{x^{(n)}}\}``, a common problem is estimating the data's probability density function ``p_X`` at given points. Normalizing Flows offer a tractable, efficient, and theoretically exact solution for this problem [^7]. 

Let ``\bm{X} \in \mathbb{R}^D`` be a random variable with an unknown probability density function ``p_X : \mathbb{R}^D \rightarrow \mathbb{R}``. With Normalizing Flows, it is possible to evaluate ``p_X(\bm{x})`` without knowing the analytical form of ``p_X``. To do this, consider ``\bm{Y} \in \mathbb{R}^D``, a random variable with a known and tractable probability density function ``p_Y : \mathbb{R}^D \rightarrow \mathbb{R}``.

Now let ``\bm{f}: \mathbb{R}^D \rightarrow \mathbb{R}^D`` be an invertible and differentiable bijection with ``\bm{y} = \bm{f}(\bm{x})``. With the [_change of variable formula_]() we can express ``p_X(\bm{x})`` as:
```math
    p_X(\bm{x}) = p_Y(\bm{f}(\bm{x})) \cdot \left| \text{det}(D\bm{f}(\bm{x}))\right|
```
Where ``\left| \text{det}(D\bm{f}(\bm{x}))\right|`` is the _absolute value_ of the determinant of the Jacobian of ``\bm{f}`` at ``\bm{x}``.
We may intuitively understand the role of this factor as the local change in volume under the transformation ``\bm{f}``.  

So the problem of the unknown form of ``p_X`` has been _transformed_ into the task of evaluating the simple density ``p_Y`` at ``\bm{f}(\bm{x})`` and calculating ``\left|\text{det}(D\bm{f}(\bm{x}))\right|``. 

As mentioned before, the evaluation of ``p_X`` in this manner is _theoretically_ exact. Via the mathematical definition, it is an analytical operation and, therefore, not subject to imprecision. In practice, though, the transformation during a Flow can only be an approximation of the theoretically ideal transformation and thus is always imperfect. Therefore, a density evaluation using this Flow is inexact. In principle, the evaluation error can be arbitrarily reduced by improving the quality of the Flow's transformation or by chaining several Flows together.

This application of Normalizing Flows in density evaluation utilizes the _forward_ transformation of already existing samples from the complex target distribution ``p_X`` and the Flow's differentiability. Another critical aspect of Normalizing Flows is their _invertibility_. It allows for the application of an _inverse_ Flow ``\bm{f}^{-1}`` to samples from a standard Normal distribution which are easily obtained. By transforming them into samples from the complex target distribution, we have effectively sampled the target distribution, which may be very challenging by conventional means.
        
### Sample Generation
Obtaining a set of data ``\mathcal{D} = \{\bm{x^{(n)}}\}`` from a particular distribution in the first place can pose a significant challenge. Here lies the second most common application of Normalizing Flows and their origin in machine learning: The generation of new samples from a target distribution. 

Given a suitable function 
```math
    \bm{g}: \mathbb{R}^D \rightarrow \mathbb{R}^D,
```
a sample ``\bm{x}`` of a complex target distribution ``p_X`` can be generated by drawing a sample ``\bm{y}`` from a simple base distribution ``p_Y`` and transforming it via ``\bm{g}`` : ``\bm{x} = \bm{g}(\bm{y})``. Also, by adapting the change of variable formula one may obtain the probability density value 
```math
    p_X(\bm{x}) = p_X(\bm{g}(\bm{y}))=  p_Y(\bm{y}) \cdot \left| \text{det}(D\bm{g}(\bm{y}))\right| ^{-1}
```
at the generated sample ``\bm{x}`` given ``p_Y(\bm{y})``.

To obtain this function ``\bm{g}``, first a forward Flow ``\bm{f}`` is generated and ``\bm{g}`` is then defined as ``\bm{g} \equiv \bm{f}^{-1}``. Given a suitable scheme, it is also possible to generate ``\bm{g}`` directly as a transformation from a simple distribution to a complex one. But this necessitates more intimate knowledge of the target distribution.


## The Transformation

The limiting factor for the success and efficiency of Normalizing Flows in the applications described previously is the method used to transform the actual data.
As research into Normalizing Flows is currently the subject of much attention, there are many different approaches to this task [^7].
        
As an example, we illustrate the method of Durkan et al., called "Neural spline Flows" [8]. They combine so-called coupling transforms and special piece-wise defined spline functions to achieve an element-wise transformation in their Normalizing Flows.


Assume the following situation:
```math
    & \bm{X} \in \mathbb{R}^D \quad \text{a random variable with} \quad \bm{X} \sim p_X : \mathbb{R}^D \rightarrow \mathbb{R}\\
    & \bm{Y} \in  \mathbb{R}^D \quad \text{a random variable with} \quad \bm{Y} \sim p_Y : \mathbb{R}^D \rightarrow \mathbb{R}\\
    & \bm{f} : \mathbb{R}^D \rightarrow \mathbb{R}^D  \quad \text{an invertible and differentiable bijection with} \quad \bm{y} = \bm{f}(\bm{x})
```
Now consider the task of transforming a sample ``\bm{x^{(0)}}`` from the complex target distribution ``p_X`` into a sample ``\bm{y^{(0)}}`` from the chosen outcome distribution ``p_Y``, and finding a transformation ``\bm{f}`` that achieves this. Durkan et al. took an element-wise approach to this, meaning each component ``x_i^{(0)}, ~ i = 1,...,D,`` of 
```math
    \bm{x^{(0)}} = 
    \begin{pmatrix}
    x_1^{(0)}   \\[4pt]
    x_2^{(0)}   \\[4pt]
    \vdots      \\[4pt]
    x_D^{(0)}   \\[4pt]
    \end{pmatrix}   
```
is transformed with an individual function
```math
    f_i : \mathbb{R} \rightarrow \mathbb{R}.
```
The function ``\bm{f}`` can thus further be written as: 
```math
    \bm{f}: \mathbb{R}^D \rightarrow \mathbb{R}^D,~ 
    \begin{pmatrix}
    x_1    \\
    x_2   \\
    \vdots \\
    x_D    \\
    \end{pmatrix}   
    \mapsto 
    \begin{pmatrix}
    f_{1}(x_1) \\
    f_{2}(x_2) \\
    \vdots \\
    f_{D}(x_D) \\
    \end{pmatrix} 
    \equiv
    \begin{pmatrix}
    y_1    \\
    y_2    \\
    \vdots \\
    y_D   \\
    \end{pmatrix}   
```

Before going into more detail about these component functions ``f_i``, we must consider an underlying challenge to this approach of element-wise transformation. 

### Coupling Transformations 

When we regard the subject of the Normalizing Flow
```math
    \bm{X} \sim p_X,
```
we can think of the separate dimensions of ``\bm{X}`` as being distributed according to ``p_X``'s respective _marginal_ distribution, ``~p_{X,i}: \mathbb{R} \rightarrow \mathbb{R},~ i= 1,...,D,`` in that dimension:
```math
    \begin{pmatrix}
    X_1  \sim p_{X,1} \\
    X_2  \sim p_{X,2} \\
    \vdots \\
    X_D  \sim p_{X,D} \\
    \end{pmatrix}  
    =
    \bm{X} \sim p_X
```
Let's assume we use element-wise transformations as described earlier to transform each dimension of ``\bm{x^{(0)}}`` to follow a one-dimensional Normal distribution: 
```math
    \bm{f}_{erroneous}(\bm{x^{(0)}}) = 
    \begin{pmatrix}
    y_1^{(0)}  \sim \mathcal{N}(0,1) \\
    y_2^{(0)}  \sim \mathcal{N}(0,1) \\
    \vdots \\
    y_D^{(0)}  \sim \mathcal{N}(0,1) \\
    \end{pmatrix}  
    =
    \bm{y^{(0)}} \nsim \mathcal{N}((0,...,0), I_D)
```
Then the marginal distributions of the result might all be Normal, but the joint distribution generally is anything but. 

This issue may be remedied by so-called _coupling transformations_. These transform an input ``\bm{x}`` by letting some of its components ``x_i`` influence the transformation of the remaining components of the sample and thus achieving a form of _coupling_ between the output's dimensions [^5].

This influence is realized by transforming a part of the input's components with functions that are characterized by parameters computed from the remaining components via a neural net. 

A Coupling Transformation ``\bm{\Phi} : \mathbb{R}^D \rightarrow \mathbb{R}^D`` maps an input ``\bm{x^{(0)}}`` to an output ``\bm{y^{(0)}}`` by first splitting ``\bm{x^{(0)}}`` into two parts. This is done by choosing a number ``d``: ``1 \leq d \leq D`` and then splitting ``\bm{x^{(0)}}`` into the first ``d-1`` and last ``D-d+1`` components:
```math
    \bm{x^{(0)}} =
    \begin{pmatrix}
    x_{1}^{(0)}    \\[4pt]
    x_{2}^{(0)}    \\[4pt]
    \vdots         \\[4pt]
    x_{D}^{(0)}    \\[4pt]
    \end{pmatrix}~\rightarrow~\bm{x^{(0,1)}} = 
    \begin{pmatrix}
    x_{1}^{(0)}    \\[4pt]
    x_{2}^{(0)}    \\[4pt]
    \vdots         \\[4pt]
    x_{d-1}^{(0)}  \\[4pt]
    \end{pmatrix}~,~\bm{x^{(0,2)}} = 
    \begin{pmatrix}
    x_{d}^{(0)}    \\[4pt]
    x_{d+1}^{(0)}  \\[4pt]
    \vdots         \\[4pt]
    x_{D}^{(0)}    \\[4pt]
    \end{pmatrix} 
```
``\bm{\Phi}`` then uses the second part ``\bm{x^{(0,2)}}`` to compute the parameters
```math
    \bm{\theta} = 
    \begin{pmatrix}
    \theta_1     \\
    \theta_2     \\
    \vdots       \\
    \theta_{d-1} \\
    \end{pmatrix}
```
that characterize the transformations ``f_{\theta_i}, i = 1,...,d-1``, of the first part ``\bm{x^{(0,1)}}``. The output ``\bm{y^{(0)}}`` is then calculated in two parts by applying the transformations ``f_{\theta_i}`` to ``\bm{x^{(0,1)}}`` and passing ``\bm{x^{(0,2)}}`` un-transformed:
```math
    \bm{y^{(0,1)}} \equiv
    \begin{pmatrix}
    f_{\theta_1}(x_{1}^{(0)})       \\[4pt]
    f_{\theta_2}(x_{2}^{(0)})       \\[4pt]
    \vdots                          \\[4pt]
    f_{\theta_{d-1}}(x_{d-1}^{(0)}) \\
    \end{pmatrix}~,~  
    \bm{y^{(0,2)}} \equiv \bm{x^{(0,2)}} = 
    \begin{pmatrix}
    x_{d}^{(0)}      \\[4pt]
    x_{d+1}^{(0)}    \\[4pt]
    \vdots           \\[4pt]
    x_{D}^{(0)}      \\
    \end{pmatrix} 
```
In summary, the Coupling Transformation ``\bm{\Phi}`` can be written as:
```math
    \bm{\Phi} : \mathbb{R}^D \rightarrow \mathbb{R}^D,~
    \begin{pmatrix}
    x_1    \\
    x_2    \\
    \vdots \\
    x_{d-1}\\
    x_{d}  \\
    \vdots \\
    x_D    \\
    \end{pmatrix}   
    \mapsto 
    \begin{pmatrix}
    y_1    \\
    y_2    \\
    \vdots \\
    y_{d-1}\\
    y_{d}  \\
    \vdots \\
    y_D    \\
    \end{pmatrix} =
    \begin{pmatrix}
    f_{\theta_1}(x_1)           \\
    f_{\theta_2}(x_2)           \\
    \vdots                      \\
    f_{\theta_{d-1}}(x_{d-1})   \\
    x_{d}                       \\
    \vdots                      \\
    x_D                         \\
    \end{pmatrix} 
```
Applying ``\bm{\Phi}`` to a sample ``\bm{x^{(0)}}`` only affects its first part ``\bm{x^{(0,1)}}``. So to transform the entire vector, the procedure has to be repeated with another coupling transformation, switching the roles of ``\bm{x^{(0,1)}}`` and ``\bm{x^{(0,2)}}``:
```math
    \text{Let} ~ \bm{\Phi}' : \mathbb{R}^D \rightarrow \mathbb{R}^D,~
    \begin{pmatrix}
    x_1    \\
    x_2    \\
    \vdots \\
    x_{d-1}\\
    x_{d}  \\
    \vdots \\
    x_D    \\
    \end{pmatrix}   
    \mapsto 
    \begin{pmatrix}
    y'_1    \\
    y'_2    \\
    \vdots \\
    y'_{d-1}\\
    y'_{d}  \\
    \vdots \\
    y'_D    \\
    \end{pmatrix} =
    \begin{pmatrix}
    x_{1}                   \\
    x_{2}                   \\
    \vdots                  \\
    x_{d-1}                 \\
    f_{\theta'_d}(x_d)      \\
    \vdots                  \\
    f_{\theta'_{D}}(x_{D})  \\
    \end{pmatrix} 
```
```math
    \bm{\Phi}' \circ \bm{\Phi} : \mathbb{R}^D \rightarrow \mathbb{R}^D,~
    \begin{pmatrix}
    x_1    \\
    x_2    \\
    \vdots \\
    x_{d-1}\\
    x_{d}  \\
    \vdots \\
    x_D    \\
    \end{pmatrix}   
    \mapsto
    \begin{pmatrix}
    y'_1    \\
    y'_2    \\
    \vdots \\
    y'_{d-1}\\
    y'_{d}  \\
    \vdots \\
    y'_D    \\
    \end{pmatrix} =
    \begin{pmatrix}
    f_{\theta_1}(x_{1})         \\
    f_{\theta_2}(x_{2})         \\
    \vdots                      \\
    f_{\theta_{d-1}}(x_{d-1})   \\
    f_{\theta'_d}(x_d)          \\
    \vdots                      \\
    f_{\theta'_{D}}(x_{D})      \\
    \end{pmatrix} 
```
Thus the composition of ``\bm{\Phi}'`` and ``\bm{\Phi}`` transforms the entire sample vector as desired while coupling the first ``d-1`` and the final ``D-d+1`` dimensions. To ensure pairwise coupling for all dimensions, the above process must be repeated with different splits to transform each dimension under the influence of every other dimension at least once.

------------

Now let us continue with Durkan et al.'s method and their choice for the function to take the place of the transformations ``f_i`` in their Normalizing Flow.

Regard the function ``\bm{f}`` 
```math
    \bm{f}: \mathbb{R}^D \rightarrow \mathbb{R}^D,~ 
    \begin{pmatrix}
    x_1    \\
    x_2   \\
    \vdots \\
    x_D    \\
    \end{pmatrix}   
    \mapsto 
    \begin{pmatrix}
    f_{1}(x_1) \\
    f_{2}(x_2) \\
    \vdots \\
    f_{D}(x_D) \\
    \end{pmatrix} 
    =
    \begin{pmatrix}
    y_1    \\
    y_2    \\
    \vdots \\
    y_D   \\
    \end{pmatrix}   
```
that transforms samples ``\bm{x} \sim p_X`` from the complex target distribution to samples ``\bm{y} \sim p_Y`` from the chosen outcome distribution. 

Now let's discuss the requirements for the component functions ``f_i`` to make ``\bm{f}`` invertible, differentiable, and tractable so we can apply the resulting Normalizing Flows in the ways discussed in sections \ref{sec:dens_eval} and \ref{sec:sample_gen}. 

The functions ``f_i`` need to

    be **Bijective**
    ``f_i`` must be bijective to enable transformations in both directions.

    have a **tractable Jacobian**
    Both the Jacobian for the forward and inverse transformations must be easily computable to make the transformation ``f`` efficient.

    be **easily invertible**
    The forward and inverse transformations have to be easily invertible so that when one is known, the other can be efficiently computed. 

In current literature, a popular choice for such functions are piece-wise defined spline functions or _splines_ for short [^7].




------------

[^1] Tabak, Esteban G. and Eric Vanden-Eijnden, *Density estimation by dual ascent of the log-likelihood* [DOI:10.1137](https://doi.org/10.1137/100783522)

[^2] Agnellit, J. P. and M. Cadeiras and Tabak, E. G. and Turnert, C. V. and E. Vanden-Eijnden, *Clustering and classification through normalizing flows in feature space* [DOI:10.1137](https://doi.org/10.1137/10078352)

[^3] PM Laurence and RJ Pignol and Esteban Tabak, *Constrained density estimation*

[^4] Oren Rippel and Ryan Prescott Adams, *High-Dimensional Probability Estimation with Deep Density Models* [arXiv:1302.5125](https://arxiv.org/abs/1302.5125)

[^5] Dinh, Laurent and Krueger, David and Bengio, Yoshua, *NICE: Non-linear Independent Components Estimation*. [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)

[^6] Danilo Jimenez Rezende and Shakir Mohamed, *Variational Inference with Normalizing Flows*. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770)

[^7] Ivan Kobyzev and Simon J.D. Prince and Marcus A. Brubaker, *Normalizing Flows: An Introduction and Review of Current Methods*. [DOI: 10.1109](https://doi.org/10.1109/TPAMI.2020.2992934)

[^8] Conor Durkan and Artur Bekasov and Iain Murray and George Papamakarios, *Neural Spline Flows*. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
