package learn;

import java.util.Arrays;

/**
 * Represents a bot with a given regression and parameters.
 * @author connor
 *
 */
public class Bot {
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
