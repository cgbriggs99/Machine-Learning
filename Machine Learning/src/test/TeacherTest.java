package test;

import static org.junit.Assert.assertTrue;

import org.junit.jupiter.api.Test;

import learn.Bot;
import learn.Gradient;
import learn.Regression;
import learn.Teacher;
import learn.loss.AbsoluteLoss;
import learn.loss.SquaredLoss;

class TeacherTest {
	private static final double[][] input1 = { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } };
	private static final double[] output1 = { 0, 1, 2, 3, 4 };
	private static final double[] output2 = { 1, 3, 5, 7, 9 };

	private Bot bot1;
	private Bot bot2;

	private static double rms(double[] arr1, double[] arr2) {
		assert (arr1.length == arr2.length);
		double sum = 0;
		for (int i = 0; i < arr1.length; i++) {
			sum += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
		}
		return (Math.sqrt(sum) / arr1.length);
	}

	@Test
	final void test() {
		bot1 = new Bot(new Regression() {

			@Override
			public double compute_regression(double[] input, double[] weight) {
				return (input[0] * weight[0] + weight[1]);
			}
		}, new double[] { 1, 0 });

		bot2 = new Bot(new Regression() {

			@Override
			public double compute_regression(double[] input, double[] weight) {
				return (input[0] * weight[0] + weight[1]);
			}
		}, new double[] { 2, 1 });
		for (int i = 0; i < 100; i++) {
			Bot bot_test_1 = Teacher.teach(new Regression() {

				@Override
				public double compute_regression(double[] input, double[] weight) {
					return (input[0] * weight[0] + weight[1]);
				}
			}, new double[] { 0, 0 }, AbsoluteLoss.getSingleton(), Gradient.getSingleton(), input1, output1, 0.001,
					0.00001);
			Bot bot_test_2 = Teacher.teach(new Regression() {

				@Override
				public double compute_regression(double[] input, double[] weight) {
					return (input[0] * weight[0] + weight[1]);
				}

			}, new double[] { 0, 0 }, AbsoluteLoss.getSingleton(), Gradient.getSingleton(), input1, output2, 0.001,
					0.00001);

			Bot bot_test_3 = Teacher.teach(new Regression() {

				@Override
				public double compute_regression(double[] input, double[] weight) {
					return (input[0] * weight[0] + weight[1]);
				}
			}, new double[] { 0, 0 }, SquaredLoss.getSingleton(), Gradient.getSingleton(), input1, output1, 0.001,
					0.00001);
			Bot bot_test_4 = Teacher.teach(new Regression() {

				@Override
				public double compute_regression(double[] input, double[] weight) {
					return (input[0] * weight[0] + weight[1]);
				}

			}, new double[] { 0, 0 }, SquaredLoss.getSingleton(), Gradient.getSingleton(), input1, output2, 0.001,
					0.00001);

			assertTrue(rms(bot_test_1.getWeight(), bot1.getWeight()) <= 0.1);
			assertTrue(rms(bot_test_2.getWeight(), bot2.getWeight()) <= 0.1);
			assertTrue(rms(bot_test_3.getWeight(), bot1.getWeight()) <= 0.1);
			assertTrue(rms(bot_test_4.getWeight(), bot2.getWeight()) <= 0.1);
		}

	}

}
