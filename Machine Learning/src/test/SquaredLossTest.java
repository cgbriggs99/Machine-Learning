package test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import learn.Regression;
import learn.loss.SquaredLoss;

class SquaredLossTest {
	
	private static final double[][] input1a = {{-1}, {0}, {1}, {2}};
	private static final double[] input1b = {5, 4, 3, 2};
	private static final double[][] input2a = {{1}, {2}, {3}, {4}, {5}};
	private static final double[] input2b = {1, 2, 3};
	private static final double output1 = 3.5;

	@Test
	final void testGetSingleton() {
		assertNotNull(SquaredLoss.getSingleton());
	}

	@Test
	final void testLoss() {
		Regression r = new Regression() {

			@Override
			public double compute_regression(double[] input, double[] weight, double bias) {
				return (input[0]);
			}
			
		};
		assertEquals(output1, SquaredLoss.getSingleton().loss(r, null, 0, input1a, input1b));
		
		assertThrows(java.lang.AssertionError.class, new Executable() {

			@Override
			public void execute() throws Throwable {
				SquaredLoss.getSingleton().loss(r, null, 0, input2a, input2b);
				
			}
			
		});
	}

}
