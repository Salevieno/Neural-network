package activationFunctions;

public class Sigmoid implements ActivationFunction
{
	
	public double f(double x)
	{
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	public double df(double x)
	{
		return x * (1 - x);
	}
	
}
