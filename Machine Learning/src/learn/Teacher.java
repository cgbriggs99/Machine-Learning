/**
 * 
 */
package learn;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Represents an abstract teacher with default implementation. Teaches an
 * algorithm how to behave.
 * 
 * @author connor
 *
 */
public class Teacher {

	protected static interface Vector {
		static double dotprod(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double sum = 0;
			for (int i = 0; i < arr1.length; i++) {
				sum += arr1[i] * arr2[i];
			}
			return (sum);
		}

		static double magnitude(double[] arr) {
			double sum = 0;
			for (double d : arr) {
				sum += d * d;
			}
			return Math.sqrt(sum);
		}

		static double[] sum(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double[] out = new double[arr1.length];
			for (int i = 0; i < arr1.length; i++) {
				out[i] = arr1[i] + arr2[i];
			}
			return out;
		}

		static double[] prod(double[] arr, double k) {
			double[] out = new double[arr.length];
			for (int i = 0; i < arr.length; i++) {
				out[i] = arr[i] * k;
			}
			return (out);
		}

		static double[] prod(double k, double[] arr) {
			return (prod(arr, k));
		}

		static double[] diff(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double[] out = new double[arr1.length];
			for (int i = 0; i < arr1.length; i++) {
				out[i] = arr1[i] - arr2[i];
			}
			return (out);
		}

		static double rms(double[] arr1, double[] arr2) {
			assert (arr1.length == arr2.length);
			double sum = 0;
			for (int i = 0; i < arr1.length; i++) {
				sum += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
			}
			return (Math.sqrt(sum) / arr1.length);
		}
	}

	public static Bot teach(Regression r, double[] weight_start, LossFunction loss, Gradient grad,
			Collection<double[]> input, Collection<Double> output, double step, double eps) {
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
				change = step;
				g1 = grad.grad(r, loss, w1, diff, input, output);
			} else {
				// If we have previous points, use those to go in an educated direction.
				g1 = grad.grad(r, loss, w1, Vector.diff(w1, w2), input, output);
				change = step;
			}

			w2 = w1;
			w1 = Vector.sum(w2, Vector.prod(change, g1));
			count++;
			//Check if the weights are close, if we are approaching a minimum, or we have gone through too many steps.
		} while ((Vector.rms(w1, w2) > eps || Vector.magnitude(g1) > eps) && count < 1 / eps);
		return (new Bot(r, w1));
	}

	public static Bot teach(Regression r, double[] weight_start, LossFunction loss, Gradient grad, double[][] input,
			double[] output, double step, double eps) {
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
