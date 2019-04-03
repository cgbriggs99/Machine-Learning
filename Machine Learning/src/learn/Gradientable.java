package learn;

/**
 * Represents a loss function that has an exact gradient.
 * @author connor
 *
 */
public interface Gradientable extends LossFunction {
	//TODO Make sure this works in context. 
	/**
	 * Returns the gradient of the loss function.
	 * @param r
	 * @param weight
	 * @param bias
	 * @param inputs
	 * @param outputs
	 * @return
	 */
	double[] loss_gradient(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs);
}
