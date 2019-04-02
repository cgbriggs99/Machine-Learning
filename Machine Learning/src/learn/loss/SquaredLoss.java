package learn.loss;

import learn.LossFunction;
import learn.Regression;

/**
 * Computes the loss function based on the average of the squared differences, normalized by the square of the size
 * rather than just the size.
 * @author connor
 *
 */
public class SquaredLoss implements LossFunction {
	private static SquaredLoss singleton = null;
	
	private SquaredLoss() {
		;
	}
	
	public static SquaredLoss getSingleton() {
		if(singleton == null) {
			singleton = new SquaredLoss();
		}
		return (singleton);
	}

	@Override
	public double loss(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs) {
		double sum = 0;
		
		assert(inputs.length == outputs.length);
		
		for(int i = 0; i < inputs.length; i++) {
			double diff = r.compute_regression(inputs[i], weight, bias) - outputs[i];
			sum += (diff * diff) / (outputs.length * outputs.length);
		}
		return (sum);
	}
}
