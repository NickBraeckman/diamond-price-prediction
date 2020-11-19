# Diamond-Price-Prediction

## Problem Definition

Predict the price of a diamond, based based on the physical characteristics.

## Dataset

Dataset is avaible on: [diamond-dataset](https://www.kaggle.com/shivam2503/diamonds)
Per diamond in dataset:

* **price** price in US dollars (326--18,823)

* **carat** weight of the diamond (0.2--5.01)

* **cut** quality of the cut (Fair, Good, Very Good, Premium, Ideal)

* **color** diamond colour, from J (worst) to D (best)

* **clarity** a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

* **x** length in mm (0--10.74)

* **y** width in mm (0--58.9)

* **z** depth in mm (0--31.8)

* **depth** total depth percentage (43--79), \begin{equation*}  D_{t,\%} = \frac{z}{mean(x, y)} = \frac{2z}{(x + y)} \end{equation*}

* **table** width of top of diamond relative to widest point (43--95)
