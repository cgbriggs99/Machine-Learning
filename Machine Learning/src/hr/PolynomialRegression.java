package hr;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

import learn.Bot;
import learn.Gradient;
import learn.Regression;
import learn.Teacher;
import learn.loss.SquaredLoss;
import learn.step.VariableStepSize;

public class PolynomialRegression {
	private static class Pair<T1, T2> {
		private T1 t1;
		private T2 t2;
		public Pair(T1 a, T2 b) {
			t1 = a;
			t2 = b;
		}
		
		public T1 getFirst() {
			return (t1);
		}
		
		public T2 getSecond() {
			return (t2);
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + ((t1 == null) ? 0 : t1.hashCode());
			result = prime * result + ((t2 == null) ? 0 : t2.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null) {
				return false;
			}
			if (!(obj instanceof Pair)) {
				return false;
			}
			@SuppressWarnings("unchecked")
			Pair<T1, T2> other = (Pair<T1, T2>) obj;
			if (t1 == null) {
				if (other.t1 != null) {
					return false;
				}
			} else if (!t1.equals(other.t1)) {
				return false;
			}
			if (t2 == null) {
				if (other.t2 != null) {
					return false;
				}
			} else if (!t2.equals(other.t2)) {
				return false;
			}
			return true;
		}
		
	}
	
	private static HashMap<Pair<Integer, Integer>, Integer> dynamic = null;
	
	private static int choose(int n, int k) {
		if(dynamic == null) {
			dynamic = new HashMap<>();
		}
		if(dynamic.containsKey(new Pair<Integer, Integer>(n, k))) {
			return (dynamic.get(new Pair<Integer, Integer>(n, k)));
		}
		if(n == k || k == 0) {
			dynamic.put(new Pair<Integer, Integer>(n, k), 1);
			return (1);
		}
		if(k > n) {
			return (0);
		}
		int out = choose(n - 1, k) + choose(n - 1, k - 1);
		dynamic.put(new Pair<Integer, Integer>(n, k), out);
		return (out);
	}

	public static void main(String[] args) throws FileNotFoundException {
		long start = System.nanoTime();
		Scanner sc;
		if (args.length == 1) {
			sc = new Scanner(new FileInputStream(args[0]));
		} else {
			sc = new Scanner(System.in);
			System.out.println("Enter the data: ");
		}

		int vars = sc.nextInt();
		int tests = sc.nextInt();
		double[][] inputs = new double[tests][vars];
		double[] outputs = new double[tests];
		for (int i = 0; i < tests; i++) {
			for (int j = 0; j < vars; j++) {
				inputs[i][j] = sc.nextDouble();
			}
			outputs[i] = sc.nextDouble();
		}

		Bot bot = Teacher.teach(new Regression() {

			@Override
			public double compute_regression(double[] input, double[] weight) {
				double out = 0;
				int[] bars = new int[input.length];
				for(int i = 0; i < choose(input.length + 4, 3); i++) {
					double prod = weight[i] * Math.pow(input[0], bars[0]);
					for(int j = 1; j < bars.length; j++) {
						prod *= Math.pow(input[j], bars[j] - bars[j - 1]);
					}
					out += prod;
					bars[0]++;
					for(int j = 0; j < bars.length - 1; j++) {
						if(bars[j] >= bars[j + 1]) {
							bars[j] = (j == 0)? 0: bars[j - 1];
							bars[j + 1]++;
						}
					}
					if(bars[bars.length - 1] >= weight.length) {
						break;
					}
				}
				return (out);
			}

		}, new double[choose(vars + 4, 3)], SquaredLoss.getSingleton(), Gradient.getSingleton(), inputs, outputs,
				new VariableStepSize(), 0.006);
		
		int checks = sc.nextInt();
		for(int i = 0; i < checks; i++) {
			double[] input = new double[vars];
			for(int j = 0; j < vars; j++) {
				input[j] = sc.nextDouble();
			}
			System.out.println(bot.compute_regression(input));
		}

		sc.close();
		long end = System.nanoTime();
		System.out.println("Took: " + (end - start) / 1000000000.0 + "s.");
	}

}
