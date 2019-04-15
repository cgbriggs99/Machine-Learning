package learn.step;

import learn.StepSize;
import utils.Vector;

public class VariableStepSize implements StepSize {

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
