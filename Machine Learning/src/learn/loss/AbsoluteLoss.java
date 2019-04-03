package learn.loss;

import learn.LossFunction;

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
	public double loss(double[] inputs, double[] outputs) {
		double sum = 0;
		
		assert(inputs.length == outputs.length);
		
		for(int i = 0; i < inputs.length; i++) {
			sum += Math.abs(inputs[i] - outputs[i]) / inputs.length;
		}
		return (sum);
	}

}
