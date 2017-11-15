---
title: Project Proposal
author: Caleb Ki & Sanjana Gupta
date: November 15, 2017
geometry: margin = 1in
fontsize: 11pt
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
	$g$ is simply the dual function given by $g(\lambda) = \inf_{x}f(x) - \sum_{i=1}^{m}\lambda_i(\mathcal{A}(x)_i + b_i),$ and $\mathcal{K}^* = \{\lambda \in \mathbb{R}^m: \langle \lambda, x \rangle \geq 0 \: \forall x \in \mathcal{K} \}$ is the dual cone. 

The dual problem is not directly solved at this step. This is because the dual function is generally not differentiable for the class of problems we are considering. Further, directly using subgradient methods is not efficient since these methods converge very slowly. In order to convert this to a problem that can be efficiently optimized, we apply a smoothing technique which modifies the primal objective function and instead solves the following problem:  
    \begin{equation} \label{smooth}
        \begin{aligned}
        & \underset{x}{\text{minimize}}
		& & f_{\mu}(x) \triangleq f(x) + \mu d(x)\\
		& \text{subject to}
		& &  \mathcal{A}(x) + b \in \mathcal{K}
		\end{aligned}
	\end{equation}
Here $\mu$ is a positive scalar and $d(x)$ is a strongly convex function called \textit{proximity function} which satisfies $d(x)\geq d(x_0) + \frac{1}{2} \|x-x_0\|^2$ for some fixed $x_0\in\mathbb{R}^n$.The dual of this problem is then as follows:
    \begin{equation} \label{smooth_dual}
          \begin{aligned}
		& \underset{\lambda}{\text{maximize}}
		& & g_{\mu}(\lambda) \\
		& \text{subject to}
		& & \lambda \in \mathcal{K}^*
           \end{aligned}
     \end{equation}
where $g_{\mu}$ is a smooth approximation of $g$. In many cases, the dual can be reduced to the following unconstrained problem:
    \begin{equation} \label{composite}
          \begin{aligned}
		& \underset{z}{\text{maximize}}
		& & -g_{sm}(z) - h(z) \\
           \end{aligned}
     \end{equation}
This is known as the \textit{composite form}. Here, $z\in\mathbb{R}^m$ is the optimization variable, $g_{sm}$ is a smooth convex function and $h$ is a non-smooth (possibly extended-value) convex function. Finally, we can efficiently solve both the smooth dual and the composite problems using optimal first order methods.

For both the problems (\ref{smooth_dual},\ref{composite}), given a sequence of step sizes $\{t_k\}$, the optimization begins with a point $\lambda_0\in\mathcal{K}^*$ and has the following updating rule respectively:
      \begin{equation} 
           \lambda_{k+1} \leftarrow \underset{\lambda\in\mathcal{K}^*}{\text{arg min}} \| \lambda_k + t_k\nabla g_{\mu} (\lambda_k) - \lambda \|_2 \\
           z_{k+1} \leftarrow \underset{y}{\text{arg min }} g_{sm}(z_k) + \langle \nabla g_{sm}(z_k), z-z_k\rangle + \frac{1}{2t_k} \| z-z_k \|^2 + h(z)
     \end{equation}
The approximate primal solution can then be recovered from the optimal value of $\lambda$ as follows:
      \begin{equation} \label{recover}
          x(\lambda) \triangleq \underset{x}{\text{arg min }} f(x) + \mu d(x) - \langle \mathcal{A}(x)+b, \lambda \rangle
     \end{equation}
Equation \ref{recover} shows that the fundamental computational primitive in this method is the efficient minimization of the sum of a linear term, a proximity function, and a non-smooth function.
         

# Proposal

The goal of this project is to port a portion of the TFOCS software package from Matlab into Python. Of the five solvers available in the package, we will implement two of them.

# References
