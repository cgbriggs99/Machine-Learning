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
	
	private volatile double[] sum;
	@Override
	public double loss(Regression r, double[] weight, double[][] inputs, double[] outputs) {
		
		assert(inputs.length == outputs.length);
		Runtime runtime = Runtime.getRuntime();
		final int cores = runtime.availableProcessors() - 1;
		sum = new double[cores];
		Thread[] ts = new Thread[cores];
		for(int i = 0; i < ts.length; i++) {
			final int blocking = i;
			ts[i] = new Thread(new Runnable() {

				@Override
				public void run() {
					sum[blocking] = 0;
					for(int i = blocking; i < inputs.length; i += cores) {
						sum[blocking] += Math.abs(r.compute_regression(inputs[i], weight) - outputs[i]);
					}
				}
				
			});
			ts[i].start();
		}
		
		for(int i = 0; i < ts.length; i++) {
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
		
		return (outsum / inputs.length);
	}

}
