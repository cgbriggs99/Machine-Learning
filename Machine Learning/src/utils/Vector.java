package utils;

public class Vector {
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
