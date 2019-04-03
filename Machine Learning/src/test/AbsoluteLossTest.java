/**
 * 
 */
package test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import learn.loss.AbsoluteLoss;

/**
 * @author connor
 *
 */
class AbsoluteLossTest {
	private static final double[] input1a = {-1, 0, 1, 2};
	private static final double[] input1b = {5, 4, 3, 2};
	private static final double[] input2a = {1, 2, 3, 4, 5};
	private static final double[] input2b = {1, 2, 3};
	private static final double output1 = 3.0;

	/**
	 * Test method for {@link learn.loss.AbsoluteLoss#getSingleton()}.
	 */
	@Test
	final void testGetSingleton() {
		assertNotNull(AbsoluteLoss.getSingleton());
	}

	/**
	 * Test method for {@link learn.loss.AbsoluteLoss#loss(learn.Regression, double[], double, double[][], double[])}.
	 */
	@Test
	final void testLoss() {
		assertEquals(output1, AbsoluteLoss.getSingleton().loss(input1a, input1b));
		
		assertThrows(java.lang.AssertionError.class, new Executable() {

			@Override
			public void execute() throws Throwable {
				AbsoluteLoss.getSingleton().loss(input2a, input2b);
				
			}
			
		});
	}

}
