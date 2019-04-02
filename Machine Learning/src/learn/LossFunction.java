
package learn;

/**
 * Represents loss functions.
 * @author connor
 *
 */
public interface LossFunction {
	
	/**
	 * Compute the loss function for a set of points.
	 * @param r	The regression model to test.
	 * @param weight The weight vector for the regression.
	 * @param bias The bias for the regression.
	 * @param inputs A list of lists of values to be passed to the regression.
	 * @param outputs A list of expected output values to compare the regression against.
	 * @return The value of the loss function for the given weights, bias, and inputs.
	 */
	double loss(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs);
}
