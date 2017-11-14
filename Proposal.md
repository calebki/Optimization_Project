---
title: Project Proposal
author: Caleb Ki & Sanjana Loser
date: November 15, 2017
geometry: margin = 1in
fontsize: 12pt
header-includes:
   - \usepackage{amsthm}
   - \usepackage{amsmath, mathtools}
bibliography: project.bib
---
# Background

Templates for first-order conic solvers, or TFOCS for short, is a software package that provides templates for efficient solvers for various convex optimization problems [@beck:etal:2011]. With problems in signal processing, machine learning, and statistics in mind, the creators developed a general framework and approach for solving convex cone problems. The standard routine within the TFOCS library is a four-step process. The notation used below is taken from the paper accompanying the software package.

The first step is to express the convex optimization problem with an equivalent conic formulation. The optimization problem should follow the form:
	\begin{equation} \label{primal}
		\begin{aligned}
		& \underset{x}{\text{minimize}}
		& & f(x) \\
		& \text{subject to}
		& & \mathcal{A}(x) + b \in \mathcal{K}
		\end{aligned}
	\end{equation}
In the problem above, $x \in \mathbb{R}^n$ is the vector we are optimizing under the convex objective function $f.$ It is important to note that $f$ does not have to be a smooth function. $\mathcal{A}$ is a linear operator from $\mathbb{R}^n$ to $\mathbb{R}^m,$ $b \in \mathbb{R}^m$ is simply a vector, and $\mathcal{K}$ is a convex cone in $\mathbb{R}^m.$

There are two main hurdles to developing efficient first-order solutions for convex optimization problems of this form. The first is that we did not restrict $f$ to the set of smooth functions, and the second is that finding feasible points under this conic constraint is often an expensive computation. These hurdles are circumvented by finding and solving the dual problem of the given convex optimization problem. The second step of the process is to turn the problem into its dual form. The dual of the form given by equation \ref{primal} is as follows:
	\begin{equation} \label{dual}
		\begin{aligned}
		& \underset{\lambda}{\text{maximize}}
		& & g(\lambda) \\
		& \text{subject to}
		& & \lambda \in \mathcal{K}^*
		\end{aligned}
	\end{equation}
	$g$ is simply the dual function given by $g(\lambda) = \inf_{x}f(x) - \sum_{i=1}^{m}\mathcal{A}(x)_i + b_i,$ and $\mathcal{K}^* = \{\lambda \in \mathbb{R}^m: \langle \lambda, x \rangle \geq 0 \: \forall x \in \mathcal{K} \}$ is the dual cone.

# Proposal

The goal of this project is to port a portion of the TFOCS software package from Matlab into Python. Of the five solvers available in the package, we will implement two of them.

# References
