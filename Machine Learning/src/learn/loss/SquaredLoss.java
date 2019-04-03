package learn.loss;

import learn.LossFunction;

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
	public double loss(double[] inputs, double[] outputs) {
		double sum = 0;
		
		assert(inputs.length == outputs.length);
		
		for(int i = 0; i < inputs.length; i++) {
			double diff = inputs[i] - outputs[i];
			sum += (diff * diff) / (outputs.length * outputs.length);
		}
		return (sum);
	}
}
