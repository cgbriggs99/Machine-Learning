package learn;

import java.util.Collection;
import java.util.Iterator;

public class Gradient {
	
	private static Gradient singleton = null;
	private Gradient() {
		;
	}
	
	public static Gradient getSingleton() {
		if(singleton == null) {
			singleton = new Gradient();
		}
		return (singleton);
	}

	/**
	 * Compute the gradient according to weight and bias.
	 * 
	 * @param r       The regression to test.
	 * @param loss    The loss function to use.
	 * @param weight  The initial weight vector.
	 * @param dweight The change vector. Each element represents a step.
	 * @param bias    The initial bias.
	 * @param dbias   The change in bias.
	 * @param input   Inputs to test against.
	 * @param output  Outputs to test against.
	 * @return A vector containing the gradient of the loss function of the
	 *         regression, with the bias gradient in the last element.
	 */
	public double[] grad(Regression r, LossFunction loss, double[] weight, double[] dweight, double[][] input,
			double[] output) {
		assert (weight.length == dweight.length);

		double[] out = new double[weight.length];

		for (int i = 0; i < weight.length; i++) {
			double[] change = weight.clone();
			change[i] += dweight[i];
			out[i] = (loss.loss(r, change, input, output) - loss.loss(r, weight, input, output)) / dweight[i];
		}
		return (out);
	}

	public double[] grad(Regression r, LossFunction loss, double[] weight, double[] dweight,
			Collection<double[]> input, Collection<Double> output) {
		assert(input.size() == output.size());
		double[] outs = new double[output.size()];
		double[][] ins = new double[output.size()][];
		Iterator<Double> out_iter = output.iterator();
		Iterator<double[]> in_iter = input.iterator();
		
		for(int i = 0; i < outs.length; i++) {
			outs[i] = out_iter.next();
			ins[i] = in_iter.next();
		}
		return (grad(r, loss, weight, dweight, ins, outs));
	}
}
