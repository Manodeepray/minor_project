Here are the class notes in an MD format:

**Principal Component of Variation**

### Key Topics

* Finding the closest point on a line to a given point
* Orthogonal directions and unit vectors
* Principal Component of Variation (PCV)
* Maximizing the projected amount onto a subspace

### Subpoints

* **Finding the closest point on a line**:
	+ Define alpha as the parameter of the line
	+ Compute the closest point using the expression: alpha = a + (p - a) dot product of (n - n1)
	+ Intuitively, the closest point should be on the line and orthogonal to the given direction
* **Orthogonal directions and unit vectors**:
	+ Define two orthogonal directions: n1 and n2
	+ Normalize the directions: ||n1|| = ||n2|| = 1
	+ Compute the dot product of n1 and n2: n1 dot n2 = 0
* **Principal Component of Variation (PCV)**:
	+ Define the subspace spanned by the orthogonal directions
	+ Find the closest point in the subspace to the given point
	+ Compute the residual: residual = ||p - a - alpha*n||^2
* **Maximizing the projected amount onto a subspace**:
	+ Define the optimization problem: max alpha subject to alpha >= 0
	+ Compute the derivative of the objective function with respect to alpha: alpha = 0
	+ Solve for alpha: alpha = (p - a) dot n / ||n||^2
* **Principal Component of Variation (PCV) interpretation**:
	+ Intuitively, the PCV is the direction that captures the most variation in the data
	+ Mathematically, the PCV is the direction that maximizes the projected amount onto the subspace

### Important Details

* The PCV is not necessarily the direction that explains the most variance in the data
* The PCV is the direction that minimizes the residual: residual = ||p - a - alpha*n||^2
* The PCV can be computed using either the maximization or minimization approach
* The PCV is a linear combination of the orthogonal directions: PCV = alpha1*n1 + alpha2*n2 + ...

I hope this helps! Let me know if you have any questions or need further clarification on any of the points.