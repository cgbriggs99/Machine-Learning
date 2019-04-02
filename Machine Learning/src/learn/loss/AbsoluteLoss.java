package learn.loss;

import learn.LossFunction;
import learn.Regression;

/**
 * Represents the loss function given by taking the average of the absolute differences.
 * @author connor
 *
 */
public class AbsoluteLoss implements LossFunction {
	private static AbsoluteLoss singleton = null;
	
	private AbsoluteLoss() {
		;
	}
	
	public static AbsoluteLoss getSingleton() {
		if(singleton == null) {
			singleton = new AbsoluteLoss();
		}
		return (singleton);
	}
	
	
	@Override
	public double loss(Regression r, double[] weight, double bias, double[][] inputs, double[] outputs) {
		double sum = 0;
		
		assert(inputs.length == outputs.length);
		
		for(int i = 0; i < inputs.length; i++) {
			sum += Math.abs(r.compute_regression(inputs[i], weight, bias) - outputs[i]) / inputs.length;
		}
		return (sum);
	}

}
