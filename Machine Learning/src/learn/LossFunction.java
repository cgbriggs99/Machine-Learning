
package learn;

import java.util.Collection;
import java.util.Iterator;

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
	 * @param inputs A list of inputs for the regression to test.
	 * @param outputs A list of outputs to compare.
	 * @return The loss function value for the given inputs.
	 */
	double loss(Regression r, double[] weight, double[][] inputs, double[] outputs);
	
	/**
	 * Same as above, but using abstract collections instead of arrays, for more extensibility.
	 * @see LossFunction.loss
	 */
	default double loss(Regression r, double[] weight, Collection<double[]> inputs, Collection<Double> outputs) {
		double[] outarray = new double[outputs.size()];
		Iterator<Double> iter = outputs.iterator();
		for(int i = 0; i < outarray.length; i++) {
			outarray[i] = iter.next().doubleValue();
		}
		
		return (loss(r, weight, (double[][]) inputs.toArray(), outarray));
	}
}
