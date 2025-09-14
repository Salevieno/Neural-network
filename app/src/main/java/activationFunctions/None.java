package activationFunctions;

public class None implements ActivationFunction
{
	
	public double f(double x)
	{
		return x;
	}
	
	public double df(double x)
	{
		return 1;
	}
	
}
