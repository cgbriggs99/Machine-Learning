package learn;

/**
 * Represents a regression.
 * @author Connor Briggs
 *
 */
public interface Regression {

	/**
	 * Compute the output value of a regression at a given point.
	 * @param input The input values to be passed.
	 * @param weight An array containing weights to be applied to each term, normally multiplicatively.
	 * @param bias A single value that is added to the whole function.
	 * @return The value of the regression at a given input value with the given weights and bias.
	 */
	double compute_regression(double[] input, double[] weight);
	
}
