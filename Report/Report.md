---
title: Implementing 2 First-Order Conic Solvers in Python
author: Caleb Ki & Sanjana Gupta
date: December 12th, 2017
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
fontsize: 11pt
header-includes:
   - \usepackage{amsthm, amssymb}
   - \usepackage{amsmath, mathtools}
   - \usepackage[utf8]{inputenc}
   - \usepackage{algorithmicx, algpseudocode, algorithm}
   - \usepackage{graphicx}
   - \usepackage{caption}
   - \usepackage{subcaption}
   - \usepackage{float}
   - \usepackage{advdate}
   - \usepackage[utf8]{inputenc}
   - \usepackage[table]{xcolor}
   - \usepackage{multirow}
   - \usepackage{array}
    \newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
 	\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}
bibliography: project.bib
output: pdf_document
abstract: In this paper we consider first-order solvers for convex cone problems. We introduce the general framework for these kinds of problems developed by the creators of the TFOCS software. The approach works by taking a conic formulation of an optimization problem, determining its dual, applying smoothing, and finally solving via a first-order method. The paper goes into detail about two of the six solvers available in TFOCS, Auslender and Teboulle’s method as well as Lam, Lu, and Monteiro’s method. In particular, the paper discusses how these algorithms use backtracking and projections to efficiently get a solution. Finally, a numerical experiment with the Lasso problem displays the performance of these two solvers in a variety of settings.
---

# Background

Templates for first-order conic solvers, or TFOCS for short, is a software package that provides efficient solvers for various convex optimization problems [@beck:etal:2011]. With problems in signal processing, machine learning, and statistics in mind, the creators developed a general framework and approach for solving convex cone problems. The standard routine within the TFOCS library is a four-step process. The notation used below is taken from the paper accompanying the software package.

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

The two main hurdles to developing efficient first-order solutions for convex optimization problems of this form are that $f$ is not restricted to the set of smooth functions and that finding feasible points under this conic constraint is often an expensive computation. These hurdles are circumvented by finding and solving the dual problem of the given convex optimization problem which leads us to the second step of the process, turning the problem into its dual form. The dual of the form given by equation \ref{primal} is as follows:
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
	\begin{equation} \label{smooth}
        \begin{aligned}
        & \underset{x}{\text{minimize}}
		& & f_{\mu}(x) \triangleq f(x) + \mu d(x)\\
		& \text{subject to}
		& &  \mathcal{A}(x) + b \in \mathcal{K}
		\end{aligned}
	\end{equation}
Here $\mu$ is a positive scalar and $d(x)$ is a strongly convex function called the \textit{proximity function} which satisfies $d(x)\geq d(x_0) + \frac{1}{2} \|x-x_0\|^2$ for some fixed $x_0\in\mathbb{R}^n$. The dual of this problem is
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
This is known as the \textit{composite form}. Here, $z\in\mathbb{R}^m$ is the optimization variable, $g_{sm}$ is a smooth convex function and $h$ is a non-smooth (possibly extended-value) convex function. Finally, we can efficiently solve both the smooth dual and the composite problems using optimal first order methods such as gradient descent.

For both the problems (\ref{smooth_dual}), (\ref{composite}), given a sequence of step sizes $\{t_k\}$, the optimization begins with a point $\lambda_0\in\mathcal{K}^*$ and has the following updating rule respectively: 
$$\lambda_{k+1} \leftarrow \underset{\lambda\in\mathcal{K}^*}{\text{arg min}} \| \lambda_k + t_k\nabla g_{\mu} (\lambda_k) - \lambda \|_2$$
$$z_{k+1} \leftarrow \underset{z} {\text{arg min }} g_{sm}(z_k) + \langle \nabla g_{sm}(z_k), z-z_k\rangle + \frac{1}{2t_k} \| z-z_k \|^2 + h(z).$$

The approximate primal solution can then be recovered from the optimal value of $\lambda$ as follows:
      \begin{equation} \label{recover}
          x(\lambda) \triangleq \underset{x}{\text{arg min }} f(x) + \mu d(x) - \langle \mathcal{A}(x)+b, \lambda \rangle
     \end{equation}
Equation \ref{recover} shows that the fundamental computational primitive in this method is the efficient minimization of the sum of a linear term, a proximity function, and a non-smooth function.
         

# Implementation

As stated previously, TFOCS, when given the correct conic formulation of an optimization problem, will apply various first-order solvers to the smoothed dual problem. For this project, we port 2 of the 6 available first-order solvers in TFOCS software package from Matlab into Python. Specifically, the 2 solvers are the default method in TFOCS, the single-projection method developed by @ausl:tebo:2006, and the dual-projection method developed by @lan:etal:2011. Given  a smooth function, the gradient of the smooth function, a nonsmooth function, and an initial point, the solver returns the optimal vector and the primal solution. 

## Motivation

For the purposes of this project, we assume that the user has expressed the optimization problem of concern in the correct specific conic form. Once the problem has been wrangled into the correct form, they can be solved through first-order methods. We assume that the optimization problem has been massaged into the following unconstrained form:
$$\text{minimize } \phi(z) \triangleq g(z) + h(z)$$
which is simply equation $\ref{composite}$ except we have flipped the signs and turned the maximization problem into a minimization problem. To be notationally consistent we fix the update rule for $z_{k+1}$ provided in the background section to
\begin{equation}\label{projection}
	z_{k+1} \leftarrow \underset{z} {\text{arg min }} g(z_k) + \langle \nabla g(z_k), z - z_k\rangle + \frac{1}{2t_k} \|z - z_k \|^2 + h(z),
\end{equation}
where $t_k$ is the step size. @beck:etal:2011 assert that to ensure global convergence the following inequality must be satisfied:
\begin{equation}\label{convcondition}
	\phi (z_{k+1}) \leq g(z_k) + \langle \nabla g(z_k), z_{k+1} - z_k \rangle + \frac{1}{2t_k} \|z_{k+1} - z_k \|^2 + h(z_{k+1}).
\end{equation}
If we assume that the gradient of $g$ satisfies, 
$$\| \nabla g(x) - \nabla g(y) \|_* \leq L \| x - y \| \: \forall x,y \in \text{dom} \phi$$
(i.e., it satifies a generalized Lipschitz criterion), then convergence condition $\ref{convcondition}$ holds for all $t_k \leq L^{-1}.$ Of course $L$ is difficult to find. Rather than finding the exact value for $L$, the solvers us an estimate $L_k$ at each iteration which is explained in detail below. Returning to the main point, all the first-order solvers discussed in @beck:etal:2011 including the 2 we implement are based on update rule $\ref{projection}$ and convergence criterion $\ref{convcondition}.$

## The Algorithm

The repeated calls to a generalized projection as described above manifests itself in the algorithm below. Again, we are assuming that the user has provided the smooth function $g$ and its gradient, the nonsmooth function $h$ and its prox function, a tolerance level for convergence, and a starting point $x_0.$

\begin{algorithm}
\caption{Auslender and Teboulle's Algorithm}\label{AT}
\begin{algorithmic}[1]
	\Require{$z_0 \in \text{dom}\theta,$ $\alpha \in (0,1],$ $\beta \in (0,1),$ $\gamma,$ tol > 0}
	\State $\bar{z}_0 \leftarrow z_0,$ $\theta_{-1} \leftarrow 1,$ $L_{-1} \leftarrow 1$
	\For{$k = 0,1,2,\dots$} 
		\State $L_k = \alpha L_{k-1}$
		\Loop
			\State $\theta_k \leftarrow 2/(1+(1+4L_k/\theta_{k-1}^2L_{k-1})^{1/2})$
			\State $y_k \leftarrow (1-\theta_k)z_k + \theta_k\bar{z}_k$
			\State $\bar{z}_{k+1} \leftarrow \text{arg min}_z \langle \nabla g(y_k), z \rangle + \frac{1}{2}\theta_kL_k\|z - \bar{z}_k\|^2 + h(z)$
			\State $z_{k+1} \leftarrow (1-\theta_k)z_k + \theta_k\bar{z}_{k+1}$
			\If{$g(y_k) - g(z_{k+1}) \geq \gamma g(z_{k+1})$}
				\State $\hat{L} \leftarrow 2(g(z_{k+1}) - g(y_k) - \langle \nabla g(y_k), z_{k+1} - y_k \rangle)/\|z_{k+1}-y_k\|_2^2$ 
			\Else
				\State $\hat{L} \leftarrow 2|\langle y_k - z_{k+1}, \nabla g(z_{k+1}) - \nabla g(y_k)  \rangle|/\|z_{k+1} - y_k\|_2^2$
			\EndIf
			\If{$L_k \geq \hat{L}$} \textbf{break} \EndIf
			\State $L_k \leftarrow \max{\{L_k/\beta, \hat{L}\}}$
		\EndLoop
		\If{$\|z_k - z_{k-1}\|/\max{\{1, \|z_k\|\}} \leq \text{tol}$} \textbf{break} \EndIf
	\EndFor	
\end{algorithmic}
\end{algorithm}

Lan, Lu, and Monteiro's modification of Nesterove's 2007 algorithm follows the same algorithm except we replace \textit{line 8} with the following call:
$$z_{k+1} \leftarrow \text{arg min}_z \langle \nabla g(y_k), z \rangle + \frac{1}{2}L_k\|z - y_k\|^2 + h(z).$$
The key difference between these two similar variants is that, Lan, Lu, and Monteiro's method (LLM) requires 2 projections where Auslender and Teboulle's (AT) only requires 1. Of course, two projections per iteration is more computationally taxing, so we only prefer the two projection method in the case that it can reduce the number of iterations significantly.

## Justification

In the following section, we go through several lines in our algorithm to justify and clarify what the solvers are actually doing. The two main components of the algorithm are how the solvers are updating the step size (backtracking) and how the solvers are updating the solution.

### Updating Step Size

We begin with a discussion about step size (\textit{lines 3,9-16}). Generally it's very difficult to calculate $L,$ and the step size $\frac{1}{L}$ is often too conservative. While the performance can be improved by reducing $L$, reducing $L$ too much can cause the algorithm to diverge. All these problems are simultaneously resolved by using \textit{backtracking}. Backtracking is a technique used to find solutions to optimization problems by building partial candidates to the solution called \textit{backtracks}. Each backtrack is dropped as soon as the algorithm realizes that it cannot be extended to a valid solution. Applying this technique to the Lipschitz constant in our problem, we estimate the global constant $L$ by $L_k$ which preserves convergence if the following inequality holds:
	\begin{equation}\label{Linequality1}
	g(z_{k+1})\leq g(y_k) + \langle \nabla g(y_k), z_{k+1}-y_k \rangle + \frac{1}{2} L_k \|z_{k+1}-y_k\|^2.
	\end{equation} 
If $g(z_{k+1})$ is very close to $g(y_k)$, equation (\ref{Linequality1}) suffers from severe cancellation errors. Generally, cancellation errors occur if $g(y_k)-g(z_{k+1})<\gamma g(z_{k+1})$ where the threshold for $\gamma$ is around $10^{-8}$ to $10^{-6}$. If we believe that the algorithm will suffer from cancellation errors, we use the following inequality to ensure convergence:
	\begin{equation}\label{Linequality2}
	|\langle y_k-z_{k+1},\nabla g(z_{k+1})- \nabla g(y_k)\rangle | \leq \frac{1}{2}L_k \|z_{k+1}-y_k\|^2.
	\end{equation}

Note that inequalities (\ref{Linequality1}) and (\ref{Linequality2}) automatically hold in the case when $L_k\geq L$, so the solver only needs to check if inequalities (\ref{Linequality1}) or (\ref{Linequality2}) hold (based on whether $g(y_k)-g(z_{k+1})<\gamma g(z_{k+1})$). If it is the case the relevant inequality holds then we need not update $L_k$ further. If however the inequality does not hold we need to increase $L_k$ until it does (i.e., backtrack until we have satisfied the convergence criterion). As part of this backtracking process we introduce $\hat{L}$ which is the smallest value of $L_k$ that satisfies the relevant inequality (\ref{Linequality1}) or (\ref{Linequality2}) at the $k$th iteration holds or not). $L_k$ is then updated as max$\{\frac{L_k}{\beta},\hat{L}\}$. Here, $\hat{L}$ is obtained by changing the inequalities (\ref{Linequality1}), (\ref{Linequality2}) to equalities and solving for $L_k$. This process of checking and updating $L_k$ is repeated until $L_k$ satisfies inequalities (\ref{Linequality1}) or (\ref{Linequality2}).

In every iteration we try to reduce the value of the Lipschitz estimate $L_k$ (\textit{line 3}). This is done to improve the performance of the algorithm and is achieved by updating $L_k=\alpha L_{k-1}$ for some fixed $\alpha\in (0,1]$. Reducing $L_k$ at each iteration can of course lead to an increased number of backtracks which we try to minimize by picking an appropriate value of $\alpha$. For these kinds of solvers, generally $\alpha = 0.9$ is used which is what we have set $\alpha$ to be. 

Intertwined with updating $L_k$ is the problem of updating $\theta_k$ (\textit{line 5}). First we establish bounds on the prediction error at the $(k+1)$st iteration as follows:
	\begin{equation}
	\phi(z_{k+1})-\phi(z^{\star})\leq\frac{1}{2}L{\theta_k}^2\|z_0-z^{\star}\|^2 \leq 2\frac{L}{k^2} \|z_0-z^{\star}\|^2
	\end{equation}
This shows that the bound on the error is directly proportional to $L_k {\theta_k}^2$. In a simple backtracking step, as we increase $L_k$ (by at least a factor of $1/\beta$), the bound on the error increases too. This can be avoided by updating $\theta_k$ along with $L_k$ in each iteration by using the following inequality which ensures that convergence is preserved:
	\begin{equation}
	\frac{L_{k+1} {\theta_{k+1}}^2 }{1-\theta_{k+1}} \geq L_{k} {\theta_{k}}^2
	\end{equation}
Solving for $\theta_{k+1}$ gives the update as in \textit{line 5} of algorithm 1.

### Updating $z_{k+1}$

The algorithm states in \textit{line 7} that to update $\bar{z}_k$  we must find the arg min of some function of the gradient of $g$ and nonsmooth function $h.$ This can be simply reduced to the proximity function for $h$ with step size $\frac{1}{L}$ evaluated at $\bar{z}_k - \frac{\nabla g(y_k)}{L_k \theta_k}$ where $y_k$ is a linear combination of $z_k$ and $\bar{z}_k$. The proximity operator \textit{prox} of a convex function $h$ at $x$ is defined as the unique solution to the following:
	\begin{equation}
	\text{prox}_{t,h}(x) = \text{arg }\underset{z}{\text{min }} h(y) + \frac{1}{2t} \|x-z\|^2
	\end{equation}
Here $t$ is the step size. The update in \textit{line 7} is then equivalent to a proximity function as follows:
	\begin{equation*}
		\begin{aligned}
		&\underset{z}{\text{arg min }} \langle \nabla g(y_k),z \rangle + \frac{L_k \theta_k}{2} \|z-\bar{z}_k\|^2 + h(z)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k\theta_k}{2}\Bigg( \frac{2}{L_k\theta_k} \langle \nabla g(y_k),z \rangle + \|z\|^2 -2\langle \bar{z}_k,z\rangle + \|\bar{z}_k\|^2\Bigg)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k\theta_k}{2}\Bigg( 2 \langle \frac{\nabla g(y_k)}{L_k\theta_k}-\bar{z}_k, z \rangle + \|z\|^2 + \|\bar{z}_k\|^2\Bigg)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k\theta_k}{2}\Bigg( 2 \langle \frac{\nabla g(y_k)}{L_k\theta_k}-y_k,z \rangle + \|z\|^2\Bigg) + \frac{L_k\theta_k}{2}\|y_k\|^2\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k\theta_k}{2}\Bigg( \|\frac{\nabla g(y_k)}{L_k\theta_k}-\bar{z}_k+z\|^2 - \| \frac{\nabla g(y_k)}{L_k\theta_k}- y_k\|^2\Bigg)  + \frac{L_k\theta_k}{2}\|\bar{z}_k\|^2\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k\theta_k}{2} \|\bar{z}_k - \frac{\nabla g(y_k)}{L_k\theta_k}-z\|^2 \\
		&= \text{prox}_{\frac{1}{L_k\theta_k},h} \Bigg(\bar{z}_k - \frac{\nabla g(y_k)}{L_k\theta_k}\Bigg)
		\end{aligned}
	\end{equation*}
This clearly holds because $\frac{L_k\theta_k}{2}\|\bar{z}_k\|^2 - \frac{L_k\theta_k}{2} \| \frac{\nabla g(y_k)}{L_k\theta_k}- y_k\|^2$ is independent of $z$ (i.e., we are able to treat it has a constant), and thus it does not affect the optimization. Therefore, we can drop this term to get the proximity function in the last line. For Auslender and Teboulle's method, we take $z_{k+1}$ to be a linear combination of $y_k$ and $bar{z}_{k+1}$ which was updated with a call to the prox function.

For the Lan, Lu, and Monteiro method, the update for $z_k$ in \textit{line 8} is not a linear combination of $y_k$ and $\bar{z}_{k+1}$, but another projection which can be computed similarly using the proximity function as follows:
	\begin{equation*}
		\begin{aligned}
		&\underset{z}{\text{arg min }} \langle \nabla g(y_k),z \rangle + \frac{L_k}{2} \|z-y_k\|^2 + h(z)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k}{2}\Bigg( \frac{2}{L_k} \langle \nabla g(y_k),z \rangle + \|z\|^2 -2\langle y_k,z\rangle + \|y_k\|^2\Bigg)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k}{2}\Bigg( 2 \langle \frac{\nabla g(y_k)}{L_k}-y_k, z \rangle + \|z\|^2 + \|y_k\|^2\Bigg)\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k}{2}\Bigg( 2 \langle \frac{\nabla g(y_k)}{L_k}-y_k,z \rangle + \|z\|^2\Bigg) + \frac{L_k}{2}\|y_k\|^2\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k}{2}\Bigg( \|\frac{g(y_k)}{L_k}-y_k+z\|^2 - \| \frac{\nabla g(y_k)}{L_k}-y_k\|^2\Bigg)  + \frac{L_k}{2}\|y_k\|^2\\
		&= \underset{z}{\text{arg min }} h(z) + \frac{L_k}{2} \|y_k - \frac{\nabla g(y_k)}{L_k}-z\|^2 \\
		&= \text{prox}_{\frac{1}{L_k},h} \Bigg(y_k - \frac{\nabla g(y_k)}{L_k}\Bigg)\label{prox1}
		\end{aligned}
	\end{equation*}
Since the term $\frac{L_k}{2} \|y_k\|^2 - \frac{L_k}{2}\| \frac{\nabla g(y_k)}{L_k}-y_k\|^2$ is independent of $z,$ it does not affect the minimization and can be dropped to obtained the proximity function as in the last line.

The final piece to the algorithm is the stopping criterion. \textit{Line 17} of the algorithm states that if the distance between sequential $z_k$'s becomes sufficiently small (i.e., less than the given tolerance) then the algorithm has converged and we have found our optimal $z.$ Usually, tolerance is taken to be $10^{-8}.$ 

# Software

Both solvers were implemented through one function called \texttt{tfocs}. The calling sequence is as follows:
$$\texttt{xstar, pstar, count = tfocs(smoothF, gradF, nonsmoothF, projectorF, x0, tol, gamma, solver)}$$
Given the smooth function of the optimization problem \texttt{smoothF}, its gradient \texttt{gradF}, the nonsmooth function of the problem \texttt{nonsmoothF}, its corresponding prox \texttt{projectorF}, and an inital point, the function \texttt{tfocs} returns the optimal vector \texttt{xstar}, the value of the function at its optimal point \texttt{pstar}, and the number of iterations the solver took to get to the solution. 

The first optional parameter is \texttt{tol} which is set to 1e-8 as the default, but can be adjusted based on the level of accuracy desired and time constraints (smaller tolerance values are more accurate but more costly). The second is \texttt{gamma} which is set at 1e-2 as the default. Smaller values of gamma lower the chance of cancellation errors but also cause the step sizes to be large and the overall convergence to the solution to be slow. Lastly, the user can decided whether to use Auslender and Teboulle's method or Lam, Lu, and Monteiro's by passing a string as the \texttt{solver} parameter. AT for Auslender and Teboulle's and LLM for Lam, Lu, and Monteiro's.

# Simulation Study

In this section we compare the performance of our two different solvers through an example. In particular, we solve the Lasso problem on different simulated datasets and evaluate the two different solvers by their error rates, number of iterations, and run times. The outcome variable $y$ was generated by 
$$ \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}, \hspace{1cm} \boldsymbol{\epsilon} \sim  \mathcal{N}_{n} (0,  \mathbf{I})$$
where the independent variables $\mathbf{X}_i$ are iid Gaussian with mean $0$ and variance $1$. Further, $\boldsymbol{\beta} = [0_{p-k},10_{k}]^T$, that is, the last k $\mathbf{X}_i$'s have a nonzero effect on $y$ whereas the the first $p-k$ $\mathbf{X}_i$'s have no relationship with $y$. 100 datasets were generated for all combinations of the following parameter settings, $n \in \{100, 500\}$, $p \in \{\frac{n}{2}, 2n \}$ and $k \in \{0.5 p,0.1 p, 0.8p\}$ (giving us 12 different settings total). We fit two Lasso models for each simulation using both AT and LLM. The unconstrained form of a Lasso problem is as follows
\begin{equation} \label{composite}
          \begin{aligned}
		& \underset{x}{\text{minimize}} 
		& &  \frac{1}{2} \|X\beta - Y\|_2^2 + \lambda \|\beta\|_1 \\
           \end{aligned}
     \end{equation}
We fit models corresponding to $\lambda = 0, \| X^T\epsilon\|_{\infty}$. $\lambda = 0$ simply gives the ordinary least squares solution whereas $\lambda = \| X^T\epsilon\|_{\infty}$ gives the correct sparse solution (theoretically). For the Lasso, the smooth function is clearly the squared 2-norm whereas the non-smooth function is the 1-norm penalty term. The arguments passed to the tfocs functions are as follows
\begin{equation}
\begin{aligned}
&\texttt{smoothF}:\: g= \frac{1}{2} \|X\beta - Y\|_2^2 \\
&\texttt{gradF}:\: \nabla g = X^T X\beta - X^T Y \\
&\texttt{nonsmoothF}:\: h = \lambda \|\beta\|_1 \\
&\texttt{projectorF}:\: \text{prox}_{t,h} = \text{softthreshold}(y,t\lambda)\text{ where }\\
		& \text{softthreshold}(y,t\lambda) = \text{sgn}(y)\max\{|y|-t\lambda\} = 
			\begin{cases} 
		      y+t\lambda & y\leq -t\lambda \\
		      0 & |y|\leq -t\lambda \\
		      y-t\lambda & y\geq t\lambda 
		   \end{cases}
\end{aligned}
\end{equation}


Table 1 summarizes the performance of the 2 techniques where the calculations are averaged over 100 simulations per setting.


\setlength{\arrayrulewidth}{0.5mm}
\setlength{\tabcolsep}{20pt}
\renewcommand{\arraystretch}{2.5}
 
\newcolumntype{s}{>{\columncolor[HTML]{AAACED}} p{0.5cm}}
 
\arrayrulecolor[HTML]{AAACED}
 
\begin{tabular}{|P{0.5cm}|P{0.5cm}|P{0.5cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|}
\hline
\multicolumn{3}{|M{4cm}|}{$\lambda = 0$}  & \multicolumn{3}{M{5.5cm}|}{AT} & \multicolumn{3}{M{5.5cm}|}{LLM}\\
\hline
n & p & k & time & error & numiters & time & error & numiters  \\
\hline
\rowcolor[HTML]{F4F6F6} 500	&	250	&	125	&	0.2529	&	0.0089	&	164.75	&	0.3120	&	0.0089	&	165.7\\
\rowcolor[HTML]{F4F6F6} 500	&	250	&	25	&	0.2608	&	0.0199	&	168.46	&	0.3230	&	0.0199	&	169.33\\
\rowcolor[HTML]{E5E8E8} 100	&	50	&	25	&	0.0247	&	0.0197	&	135.78	&	0.0290	&	0.0197	&	135.91\\
\rowcolor[HTML]{E5E8E8} 100	&	50	&	5	&	0.0243	&	0.0441	&	133.86	&	0.0286	&	0.0441	&	133.87\\
\rowcolor[HTML]{F4F6F6} 500	&	1000	&	500	&	1.5553	&	0.7076	&	203.51	&	1.9363	&	0.7076	&	203.82\\
\rowcolor[HTML]{F4F6F6} 500	&	1000	&	100	&	1.2353	&	0.7094	&	197.23	&	1.5387	&	0.7094	&	197.9\\
\rowcolor[HTML]{E5E8E8} 100	&	200	&	100	&	0.0641	&	0.7071	&	160.85	&	0.0765	&	0.7071	&	160.4\\
\rowcolor[HTML]{E5E8E8} 100	&	200	&	20	&	0.0594	&	0.7071	&	152.73	&	0.0722	&	0.7071	&	153.17\\
\hline
\end{tabular}
\\
\\
\begin{tabular}{|P{0.5cm}|P{0.5cm}|P{0.5cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|}
\hline
\multicolumn{3}{|M{4cm}|}{$\lambda = \| X^T\epsilon\|_{\infty}$}  & \multicolumn{3}{M{5.5cm}|}{AT} & \multicolumn{3}{M{5.5cm}|}{LLM}\\
\hline
n & p & k & time & error & numiters & time & error & numiters  \\
\hline
\rowcolor[HTML]{F4F6F6}500	&	250	&	125	&	0.2058	&	0.0262	&	132.4	&	0.1786	&	0.0230	&	94.03\\
\rowcolor[HTML]{F4F6F6}500	&	250	&	25	&	0.2486	&	0.0170	&	157.04	&	0.1358	&	0.0155	&	70.83\\
\rowcolor[HTML]{E5E8E8}100	&	50	&	25	&	0.0197	&	0.0447	&	108.16	&	0.0176	&	0.0435	&	81.38\\
\rowcolor[HTML]{E5E8E8}100	&	50	&	5	&	0.0228	&	0.0284	&	122.97	&	0.0134	&	0.0286	&	61.62\\
\rowcolor[HTML]{F4F6F6}500	&	1000	&	500	&	3.5022	&	0.8875	&	464.76	&	4.9835	&	0.8996	&	524.07\\
\rowcolor[HTML]{F4F6F6}500	&	1000	&	100	&	2.3142	&	0.0467	&	368.68	&	1.0246	&	0.0277	&	131.85\\
\rowcolor[HTML]{E5E8E8}100	&	200	&	100	&	0.1347	&	0.8856	&	345.91	&	0.1736	&	0.8921	&	364.32\\
\rowcolor[HTML]{E5E8E8}100	&	200	&	20	&	0.0974	&	0.0683	&	246.88	&	0.0494	&	0.0570	&	103.4\\
\hline
\end{tabular}
\\
\\
\begin{tabular}{|P{0.5cm}|P{0.5cm}|P{0.5cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|P{0.9cm}|}
\hline
\multicolumn{3}{|M{4cm}|}{$\lambda = \| X^T\epsilon\|_{\infty}$}  & \multicolumn{3}{M{5.5cm}|}{AT} & \multicolumn{3}{M{5.5cm}|}{LLM}\\
\hline
n & p & k & time & error & numiters & time & error & numiters  \\
\hline
\rowcolor[HTML]{F4F6F6}500500	&	1000	&	800	&	3.0070	&	0.9313	&	505.05	&	4.4929	&	0.9447	&	594.73\\
\rowcolor[HTML]{F4F6F6}500100	&	50	&	40	&	0.0182	&	0.0593	&	110.33	&	0.0206	&	0.0588	&	105.63\\
\rowcolor[HTML]{E5E8E8}500	&	250	&	200	&	0.1939	&	0.0331	&	127.04	&	0.2292	&	0.0315	&	123.01\\
\rowcolor[HTML]{E5E8E8}100	&	200	&	160	&	0.1482	&	0.9331	&	374.36	&	0.1951	&	0.9405	&	406.09\\
\hline
\end{tabular}




The top half of each table corresponds to overdetermined cases whereas the bottom half corresponds to undetdetermined ones. As we can see, neither one of the methods performs better than the other for every scenario. In the case where $\lambda=0$ (i.e. ordinary least squares), we observe that AT performs better than LLM. While the errors for both the methods are almost the same, the average number of iterations are similar which resuls in AT converging faster as it only computes a single projection in every iteration. In the overdetermined case, LLM performs better than AT in terms of both run time and error with the performance being significantly better for very sparse data (ratio of k to p is 1:10). In the underdetermined case, however, we observe that AT outperforms LLM when the ratio of k to p is 1:2 and 4:5. It has less iterations (making it almost 1.4 times as fast as LLM) and has lower error rates. This trend is particularly visible in the underdetermined case for $\lambda=0$ as well. 

While there are certain settings as the one described above where AT outperforms LLM, we found that in general LLM outperformed AT in most of the settings for the Lasso problem. We would like to point here that the runtime for LLM was much lower than that for AT and the errors were lower too. Summarizing, we observe that 
\begin{itemize}
\item AT outperforms LLM in certain cases, such as the underdetermined cases when the \textbf{ratio of k to p is 1:2 and 4:5}.
\item Despite being more computationally heavy, LLM performed atleast twice as fast for all the other datasets when $\lambda \neq 0$ in the Lasso problem considered here.
\end{itemize}

# References

