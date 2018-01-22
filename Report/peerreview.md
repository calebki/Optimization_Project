---
title: Peer Review
date: December 11th, 2017
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
fontsize: 12pt
header-includes:
   - \usepackage{amsthm}
   - \usepackage{amsmath, mathtools}
   - \usepackage[utf8]{inputenc}
bibliography: project.bib
output: pdf_document
---

# Summary

In this project report, the authors introduce a convex optimization problem in the field of medical imaging. In particular, they discuss Positron Emission Tomography (PET) which can be used to uncover metabolic changes and to discriminate healthy tissue from diseased tissue. After the convex optimization problem is introduced, the focus briefly shifts to existing methods to solve the problem such as the EM algorithm. According to the authors, the EM algorithm and other methods suffer from what they call the positive bias problem. As an attempt to alleviate the problem, they introduce two different approaches to the problem: mirror descent and Alternating Direction Method of Multipliers (ADDM).These two methods allow for negative image values which could assuage the positive bias problem.

To test their two methods, the authors simulated PET imaging using the extended cardiac-torso phantom. They found that the EM algorithm performs better than their 2 proposed mehtods in the high count setting, but ADDM performs better in the low count setting. They conclude, that in the low count setting which is where the positive bias problem exists, ADDM displays the best performance  activity recovery.

# Comments & Suggestions

I think the authors do a great job introducing the convex optimization problem and discussing the 2 methods they are proposing to use instead of the EM algorithm. The attention to detail makes it easy to follow. One thing that is unclear and could be fleshed out is why the EM algorithm truncates negative values and why must we propose entirely new methods rather than a variant of the EM algorithm. On the other hand, I found that the discussion of the simulation and results a little lacking. While the authors mention that ADDM performs better than the EM algorithm in certain settings, the paper could really benefit from a longer discussion on why this is only the case in the lower count setting (there is only one sentence that alludes to this). Discussion about the difference between lower and higher count settings would also be helpful. Further, discussion about the performance of the MD algorithm is very brief and could use more details as well. Lastly, another minor point is that the authors display 3 different settings but only discuss A (high count setting) and C (low count setting). I would suggest either removing setting B or providing some sort of discussion of the methods in that setting.
