package learn.loss;

import learn.LossFunction;
import learn.Regression;

/**
 * Computes the loss function based on the average of the squared differences.
 * 
 * @author connor
 *
 */
public class SquaredLoss implements LossFunction {
	private static SquaredLoss singleton = null;

	private SquaredLoss() {
		;
	}

	public static SquaredLoss getSingleton() {
		if (singleton == null) {
			singleton = new SquaredLoss();
		}
		return (singleton);
	}

	private volatile double[] sum;

	@Override
	public double loss(Regression r, double[] weight, double[][] inputs, double[] outputs) {
		assert (inputs.length == outputs.length);

		Runtime runtime = Runtime.getRuntime();
		final int cores = runtime.availableProcessors() - 1;
		sum = new double[cores];
		Thread[] ts = new Thread[cores];
		for (int i = 0; i < ts.length; i++) {
			final int blocking = i;
			ts[i] = new Thread(new Runnable() {

				@Override
				public void run() {
					sum[blocking] = 0;
					for (int i = blocking; i < inputs.length; i += cores) {
						sum[blocking] += Math.pow(r.compute_regression(inputs[i], weight) - outputs[i], 2);
					}
				}

			});
			ts[i].start();
		}

		for (int i = 0; i < ts.length; i++) {
			try {
				ts[i].join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		double outsum = 0;
		for(double d : sum) {
			outsum += d;
		}
		return (outsum / (inputs.length * inputs.length));
	}
}
