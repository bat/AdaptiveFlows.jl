# AdaptiveFlows.jl

AdaptiveFlows.jl provides a framework for working with Normalizing Flows, a method from machine learning that transforms complex probability distributions into simpler ones, often a $D$-dimensional standard Normal distribution. This transformation is achieved through a series of invertible and differentiable mappings.

The package is particularly useful when dealing with real-world data that follow challenging probability distributions, making traditional statistical methods difficult or impossible to apply. By shifting the focus from the complex distribution itself to finding a transformation from it to a simple distribution, Normalizing Flows can alleviate these difficulties.

This package offers functionality for both forward and inverse transformations. This means that not only can it transform samples drawn from a complex distribution so that the transformed samples follow a simple distribution, but it can also transform samples drawn from a simple distribution so that the transformed samples follow a complex target distribution.

With these functionalities, AdaptiveFlows.jl provides tools for density evaluation and sampling of a target distribution, making it a versatile tool for tasks such as clustering and classification, density estimation, and variational inference.