
package learn;

/**
 * Represents loss functions.
 * @author connor
 *
 */
public interface LossFunction {
	
	/**
	 * Computes the value of the loss function for a given regression, weight, and bias on a list of given inputs,
	 * compared to the outputs.
	 * @param r The regression to test.
	 * @param weight The weight vector to pass to the regression.
	 * @param bias The bias value to pass to the regression.
	 * @param inputs A list of inputs for the regression to test.
	 * @param outputs A list of outputs to compare.
	 * @return The loss function value for the given inputs.
	 */
	double loss(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs);
}
