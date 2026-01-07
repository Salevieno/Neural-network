package activationFunctions;

public class ReLu implements ActivationFunction
{ 	
	public double f(double x)
	{
		return 0 < x ? x : 0;
	}
	
	public double df(double x)
	{
		return 0 < x ? 1 : 0;
	}
}
