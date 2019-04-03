package learn.step;

import learn.StepSize;

public class ConstantStepSize implements StepSize {
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
