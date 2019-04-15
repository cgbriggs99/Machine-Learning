package hackerrank;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

public class HackerrankML {
	public static class Bot {
		protected Regression regression;
		protected double[] weight;
		
		public Bot(Regression r, int weight_size) {
			regression = r;
			weight = new double[weight_size];
		}
		
		public Bot(Regression r, double[] weights) {
			weight = weights.clone();
			regression = r;
		}

		public synchronized Regression getRegression() {
			return regression;
		}

		public synchronized void setRegression(Regression regression) {
			this.regression = regression;
		}

		public synchronized double[] getWeight() {
			return weight;
		}

		public synchronized void setWeight(double[] weight) {
			assert(this.weight.length == weight.length);
			this.weight = weight.clone();
		}

		public double compute_regression(double[] input) {
			return regression.compute_regression(input, weight);
		}

		@Override
		public String toString() {
			return "Bot [regression=" + regression + ", weight=" + Arrays.toString(weight) + "]";
		}
		
	}
	
	public static class Gradient {
		
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
	
	public static interface Gradientable extends LossFunction {
		//TODO Make sure this works in context. 
		/**
		 * Returns the gradient of the loss function.
		 * @param r
		 * @param weight
		 * @param inputs
		 * @param outputs
		 * @return
		 */
		double[] loss_gradient(Regression r, double[] weight, double[][] inputs, double[] outputs);
	}
	
	public static interface LossFunction {
		
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
	
	public static interface Regression {

		/**
		 * Compute the output value of a regression at a given point.
		 * @param input The input values to be passed.
		 * @param weight An array containing weights to be applied to each term, normally multiplicatively.
		 * @param bias A single value that is added to the whole function.
		 * @return The value of the regression at a given input value with the given weights and bias.
		 */
		double compute_regression(double[] input, double[] weight);
		
	}
	
	public static interface StepSize {

		double stepsize(double[] weight1, double[] weight2, double[] grad1, double[] grad2);
	}
	
	public static class Teacher {

		public static Bot teach(Regression r, double[] weight_start, LossFunction loss, Gradient grad,
				Collection<double[]> input, Collection<Double> output, StepSize step, double eps) {
			assert(input.size() == output.size());
			double[] w1, w2;
			double[] g1, g2;
			int count = 0;
			w2 = null;
			w1 = weight_start.clone();
			g2 = null;
			g1 = null;
			do {
				g2 = g1;
				double change = 0;
				if (g2 == null || w2 == null) {
					// If no previous points have been found, go a random direction.
					double[] diff = new double[w1.length];
					for (int i = 0; i < w1.length; i++) {
						diff[i] = 2 * Math.random() - 1;
					}
					change = 0.1;
					g1 = grad.grad(r, loss, w1, diff, input, output);
				} else {
					// If we have previous points, use those to go in an educated direction.
					g1 = grad.grad(r, loss, w1, Vector.diff(w1, w2), input, output);
					for(double g : g1) {
						if(!Double.isFinite(g)) {
							//This is because, as we approach the minimum, there is more likely to be a zero in the
							//denominator of the gradient expression. In that case, we are probably close enough to 
							//the answer.
							if(Vector.magnitude(g2) < 10 * eps || Vector.rms(w1, w2) < 10 * eps) {
								return (new Bot(r, w1));
							}
							throw(new java.lang.ArithmeticException());
						}
					}
					change = step.stepsize(w1, w2, g1, g2);
					if(!Double.isFinite(change)) {
						throw(new java.lang.ArithmeticException());
					}
				}

				w2 = w1;
				w1 = Vector.diff(w2, Vector.prod(change, g1));
				count++;
				//Check if the weights are close, if we are approaching a minimum, or we have gone through too many steps.
			} while ((Vector.rms(w1, w2) > eps || (Vector.magnitude(g1) > eps && Vector.magnitude(g2) > eps)) && count < 1 / eps);
			return (new Bot(r, w1));
		}

		public static Bot teach(Regression r, double[] weight_start, LossFunction loss, Gradient grad, double[][] input,
				double[] output, StepSize step, double eps) {
			assert(input.length == output.length);
			ArrayList<double[]> ins = new ArrayList<double[]>();
			ArrayList<Double> outs = new ArrayList<>();
			for(int i = 0; i < input.length; i++) {
				ins.add(input[i]);
				outs.add(output[i]);
			}
			return (teach(r, weight_start, loss, grad, ins, outs, step, eps));
		}
	}
	
	public static class AbsoluteLoss implements LossFunction {
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
	
	public static class SquaredLoss implements LossFunction {
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
	
	public static class ConstantStepSize implements StepSize {
		double step;
		public ConstantStepSize(double step) {
			this.step = step;
		}
		@Override
		public double stepsize(double[] weight1, double[] weight2, double[] grad1, double[] grad2) {
			return (step);
		}
		public synchronized double getStep() {
			return step;
		}
		public synchronized void setStep(double step) {
			this.step = step;
		}
		
	}
	
	public static class VariableStepSize implements StepSize {

		private double scale;

		public VariableStepSize(double scale) {
			this.scale = scale;
		}

		public VariableStepSize() {
			this.scale = 1;
		}

		@Override
		public double stepsize(double[] weight1, double[] weight2, double[] grad1, double[] grad2) {
			if (Vector.magnitude(Vector.diff(grad1, grad2)) == 0) {
				return (scale * Math.abs(Vector.dotprod(Vector.diff(weight1, weight2), grad1)) / Vector.dotprod(grad1, grad1));
			}
			return (scale * Math.abs(Vector.dotprod(Vector.diff(weight1, weight2), Vector.diff(grad1, grad2)))
					/ Vector.dotprod(Vector.diff(grad1, grad2), Vector.diff(grad1, grad2)));
		}

	}
	
	public static class Vector {
		public static double dotprod(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double sum = 0;
			for (int i = 0; i < arr1.length; i++) {
				sum += arr1[i] * arr2[i];
			}
			return (sum);
		}

		public static double magnitude(double[] arr) {
			double sum = 0;
			for (double d : arr) {
				sum += d * d;
			}
			return Math.sqrt(sum);
		}

		public static double[] sum(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double[] out = new double[arr1.length];
			for (int i = 0; i < arr1.length; i++) {
				out[i] = arr1[i] + arr2[i];
			}
			return out;
		}

		public static double[] prod(double[] arr, double k) {
			double[] out = new double[arr.length];
			for (int i = 0; i < arr.length; i++) {
				out[i] = arr[i] * k;
			}
			return (out);
		}

		public static double[] prod(double k, double[] arr) {
			return (prod(arr, k));
		}

		public static double[] diff(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double[] out = new double[arr1.length];
			for (int i = 0; i < arr1.length; i++) {
				out[i] = arr1[i] - arr2[i];
			}
			return (out);
		}

		public static double rms(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double sum = 0;
			for (int i = 0; i < arr1.length; i++) {
				sum += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
			}
			return (Math.sqrt(sum) / arr1.length);
		}
	}
}
