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
bibliography: project.bib
output: pdf_document
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
$$z_{k+1} \leftarrow \underset{z}{\text{arg min }} g_{sm}(z_k) + \langle \nabla g_{sm}(z_k), z-z_k\rangle + \frac{1}{2t_k} \| z-z_k \|^2 + h(z)$$

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
	z_{k+1} \leftarrow \underset{z}\text{arg min } g(z_k) + \langle \nabla g(z_k), z - z_k\rangle + \frac{1}{2t_k} \|z - z_k \|^2 + h(z),
\end{equation}
where $t_k$ is the step size. @beck:etal:2011 assert that to ensure global convergence the following inequality must be satisfied:
\begin{equation}\label{convcondition}
	\phi (z_{k+1}) \leq g(z_k) + \langle \nabla g(z_k), z_{k+1} - z_k \rangle + \frac{1}{2t_k} \|z_{k+1} - z_k \|^2 + h(z_{k+1}).
\end{equation}
If we assume that the gradient of $g$ satisfies, 
$$\| \nabla g(x) - \nabla g(y) \|_* \leq L \| x - y \| \: \forall x,y \in \text{dom} \phi$$
(i.e., it satifies a generalized Lipschitz criterion), then convergence condition $\ref{convcondition}$ holds for all $t_k \leq L^{-1}.$ Of course $L$ is difficult to find. Rather than finding the except value for $L$, the solvers us an estimate $L_k$ at each iteration which is explained in detail below. Returning to the main point, all the first-order solvers discussed in @beck:etal:2011 including the 2 we implement are based on update rule $\ref{projection}$ and convergence criterion $\ref{convcondition}.$

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
The key difference between these two similar variants is that, Lan, Lu, and Monteiro's method requires 2 projections where Auslender and Teboulle's only requires 1. Of course, two projections per iteration is more computationally taxing, so we only prefer the two projection method in the case that it can reduce the number of iterations significantly.

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

For the Lan, Lu, and Monteiro method, the update for $z_k$ in \textit{line 8} is not a linear combination of $y_k$ and $\bar{z}_{k+1}, but another projection which can be computed similarly using the proximity function as follows:
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

# Simulation Study

In this section we compare the performance of our two different solvers through an example. In particular, we solve the Lasso problem on different simulated datasets and evaluate the two different solvers by their error rates, number of iterations, and run times. The outcome variable $y$ was generated by 
$$ \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}, \hspace{1cm} \boldsymbol{\epsilon} \sim  \mathcal{N}_{n} (0,  \mathbf{I})$$
where the independent variables $\mathbf{X}_i$ are iid Gaussian with mean $0$ and variance $1$. Further, $\boldsymbol{\beta} = [0_{p-k},10_{k}]^T$, that is, the last k $\mathbf{X}_i$'s have a nonzero effect on $y$ whereas the the first $p-k$ $\mathbf{X}_i$'s have no relationship with $y$. 100 datasets were generated for all combinations of the following parameter settings, $n \in \{100, 500\}$, $p \in \{\frac{n}{2}, 2n \}$ and $k \in \{0.5 p,0.1 p\}$ (giving us 8 different settings total). We fit Lasso models for each simulation using both AT and LLM. Table 1 summarizes the performance of the 2 techniques; the calculations are averaged over the 100 simulations per setting.



# References
