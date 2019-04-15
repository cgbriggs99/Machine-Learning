/**
 * 
 */
package learn;

import java.util.ArrayList;
import java.util.Collection;

import utils.Vector;

/**
 * Represents an abstract teacher with default implementation. Teaches an
 * algorithm how to behave.
 * 
 * @author connor
 *
 */
public class Teacher {

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
