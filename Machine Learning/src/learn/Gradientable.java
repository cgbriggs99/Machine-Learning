package learn;

/**
 * Represents a loss function that has an exact gradient.
 * @author connor
 *
 */
public interface Gradientable extends LossFunction {
	double[] loss_gradient(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs);
}
