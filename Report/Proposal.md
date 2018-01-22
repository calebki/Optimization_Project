---
title: Project Proposal
author: Caleb Ki & Sanjana Gupta
date: November 15, 2017
geometry: "left=2cm,right=2cm,top=2cm,bottom=2cm"
fontsize: 11pt
header-includes:
   - \usepackage{amsthm}
   - \usepackage{amsmath, mathtools}
   - \usepackage[utf8]{inputenc}
bibliography: project.bib
output: pdf_document
---
# Background

\vspace*{-3mm}
Templates for first-order conic solvers, or TFOCS for short, is a software package that provides efficient solvers for various convex optimization problems [@beck:etal:2011]. With problems in signal processing, machine learning, and statistics in mind, the creators developed a general framework and approach for solving convex cone problems. The standard routine within the TFOCS library is a four-step process. The notation used below is taken from the paper accompanying the software package.

The first step is to express the convex optimization problem with an equivalent conic formulation. The optimization problem should follow the form:
	\vspace*{-3mm}
	\begin{equation} \label{primal}
		\begin{aligned}
		& \underset{x}{\text{minimize}}
		& & f(x) \\
		& \text{subject to}
		& & \mathcal{A}(x) + b \in \mathcal{K}
		\end{aligned}
	\end{equation}
In the problem above, $x \in \mathbb{R}^n$ is the vector we are optimizing under the convex objective function $f.$ It is important to note that $f$ does not have to be a smooth function. $\mathcal{A}$ is a linear operator from $\mathbb{R}^n$ to $\mathbb{R}^m,$ $b \in \mathbb{R}^m$ is simply a vector, and $\mathcal{K}$ is a convex cone in $\mathbb{R}^m.$

The two main hurdles to developing efficient first-order solutions for convex optimization problems of this form are that $f$ is not restricted to the set of smooth functions and that finding feasible points under this conic constraint is often an expensive computation. These hurdles are circumvented by finding and solving the dual problem of the given convex optimization problem which leads us to the second step of the process, turning the problem into its dual form. The dual of the form given by equation \ref{primal} is as follows:
	\vspace*{-3mm}
	\begin{equation} \label{dual}
		\begin{aligned}
		& \underset{\lambda}{\text{maximize}}
		& & g(\lambda) \\
		& \text{subject to}
		& & \lambda \in \mathcal{K}^*
		\end{aligned}
	\end{equation}
	$g$ is simply the dual function given by $g(\lambda) = \inf_{x}f(x) - \sum_{i=1}^{m} \langle \mathcal{A}(x) + b, \lambda \rangle,$ and $\mathcal{K}^* = \{\lambda \in \mathbb{R}^m: \langle \lambda, x \rangle \geq 0 \: \forall x \in \mathcal{K} \}$ is the dual cone. 

The dual problem is not directly solved at this step. This is because the dual function is generally not differentiable for the class of problems we are considering. Further, directly using subgradient methods is not efficient since these methods converge very slowly. In order to convert this to a problem that can be efficiently optimized, we apply a smoothing technique which modifies the primal objective function and instead solves the following problem:  
    \vspace*{-3mm}
	\begin{equation} \label{smooth}
        \begin{aligned}
        & \underset{x}{\text{minimize}}
		& & f_{\mu}(x) \triangleq f(x) + \mu d(x)\\
		& \text{subject to}
		& &  \mathcal{A}(x) + b \in \mathcal{K}
		\end{aligned}
	\end{equation}
Here $\mu$ is a positive scalar and $d(x)$ is a strongly convex function called the \textit{proximity function} which satisfies $d(x)\geq d(x_0) + \frac{1}{2} \|x-x_0\|^2$ for some fixed $x_0\in\mathbb{R}^n$. The dual of this problem is
	\vspace*{-3mm}
    \begin{equation} \label{smooth_dual}
          \begin{aligned}
		& \underset{\lambda}{\text{maximize}}
		& & g_{\mu}(\lambda) \\
		& \text{subject to}
		& & \lambda \in \mathcal{K}^*
           \end{aligned}
     \end{equation}
where $g_{\mu}$ is a smooth approximation of $g$. In many cases, the dual can be reduced to the following unconstrained problem:
    \vspace*{-3mm}
	\begin{equation} \label{composite}
          \begin{aligned}
		& \underset{z}{\text{maximize}}
		& & -g_{sm}(z) - h(z) \\
           \end{aligned}
     \end{equation}
This is known as the \textit{composite form}. Here, $z\in\mathbb{R}^m$ is the optimization variable, $g_{sm}$ is a smooth convex function and $h$ is a non-smooth (possibly extended-value) convex function. Finally, we can efficiently solve both the smooth dual and the composite problems using optimal first order methods such as gradient descent.

For both the problems (\ref{smooth_dual}), (\ref{composite}), given a sequence of step sizes $\{t_k\}$, the optimization begins with a point $\lambda_0\in\mathcal{K}^*$ and has the following updating rule respectively: 
$$\lambda_{k+1} \leftarrow \underset{\lambda\in\mathcal{K}^*}{\text{arg min}} \| \lambda_k + t_k\nabla g_{\mu} (\lambda_k) - \lambda \|_2$$
$$z_{k+1} \leftarrow \underset{y}{\text{arg min }} g_{sm}(z_k) + \langle \nabla g_{sm}(z_k), z-z_k\rangle + \frac{1}{2t_k} \| z-z_k \|^2 + h(z)$$

The approximate primal solution can then be recovered from the optimal value of $\lambda$ as follows:
      \begin{equation} \label{recover}
          x(\lambda) \triangleq \underset{x}{\text{arg min }} f(x) + \mu d(x) - \langle \mathcal{A}(x)+b, \lambda \rangle
     \end{equation}
Equation \ref{recover} shows that the fundamental computational primitive in this method is the efficient minimization of the sum of a linear term, a proximity function, and a non-smooth function.
         

# Proposal

To summarize the background, TFOCS, when given the correct conic formulation of an optimization problem will apply various first-order solvers to the smoothed dual problem. The goal of this project is to port a portion of the TFOCS software package from Matlab into Python. TFOCS implements six different first-order methods. The default is the single-projection method developed by @ausl:tebo:2006. For our project we will be implementing the default algorithm as well as the dual-projection method developed by @lan:etal:2011. In particular, we would like our solvers to accept a pythonic version of the calling sequence found in the user manual written by @beck:etal:2014 which is: \texttt{[x, out] = tfoc(smoothF, affineF, nonsmoothF, x0)}. Given  a smooth function $\texttt{smoothF}$, an affine form specification $\texttt{affineF}$, a nonsmooth function $\texttt{nonsmoothF}$, and an initial point $\texttt{x0}$, the solver would return the optimal vector $\texttt{x}$ and the primal solution $\texttt{out}$. 

# References
