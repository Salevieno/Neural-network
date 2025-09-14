package neural.network;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

import activationFunctions.ActivationFunction;

public abstract class ANN
{

    protected int iter ;
    protected int qtdIter ;
	protected double learningRate ;
    protected boolean biasIsActive ;
	protected int[] qtdNeuronsInLayer;
	protected int qtdLayers ;
	protected Results results ;
	protected Mode mode ;
	protected final ActivationFunction act ;

	protected InfoPanel infoPanel ;
	protected ANNPanel annPanel ;

	protected final Data trainingData = new Data("input.json") ;
	protected static final int STD_MAX_ITERATIONS = 1 ;

	public ANN(Point topLeftPos, int[] qtdNeuronsInLayer, ActivationFunction act)
	{
        this.iter = 0 ;
        this.qtdIter = STD_MAX_ITERATIONS ;
		this.infoPanel = new InfoPanel(new Point(topLeftPos.x, topLeftPos.y)) ;
		this.annPanel = new ANNPanel(new Point(topLeftPos.x + 10 + infoPanel.size.width, topLeftPos.y)) ;
		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		this.qtdLayers = qtdNeuronsInLayer.length ;
		this.results = new Results() ;
		this.mode = Mode.train ;
		this.act = act ;
	}

	public abstract void forwardPropagation(List<Double> input) ;
    public abstract void train(List<DataPoint> trainingDataPoints) ;
    public abstract void test(List<DataPoint> trainingDataPoints) ;
    public abstract List<Double> use(List<Double> inputs) ;
	public abstract void display() ;

	protected static double updateLRate(double lRate, double error, double min, double max) { return Math.max(Math.min(lRate - 0.01 * (error - 0.6), max), min) ;}

	protected static List<Double> extractAllInputData(List<DataPoint> trainingData)
	{
		List<Double> allInputData = new ArrayList<>();
		for (DataPoint dataPoint : trainingData)
		{
			allInputData.addAll(dataPoint.getInputs());
		}
		return allInputData;
	}

	protected static List<Double> extractAllTargetData(List<DataPoint> trainingData)
	{
		List<Double> allInputData = new ArrayList<>();
		for (DataPoint dataPoint : trainingData)
		{
			allInputData.addAll(dataPoint.getTargets());
		}
		return allInputData;
	}
	
	protected double calcOutputErrorPropagatedToLastLayer(double target, double output, double neuronInput) { return -(target - output) * act.df(neuronInput) ;}
	protected double calcError(double target, double output) { return Math.pow(target - output, 2) * 1.0 / 2.0 ;}
	
	public double calcAvrErrorPerc()
	{
		double error = 0;
		for (int t = 0; t <= trainingData.getDataPoints().size() - 1; t += 1)
		{
			List<Double> inputs = trainingData.getDataPoints().get(t).getInputs() ;
			List<Double> targets = trainingData.getDataPoints().get(t).getTargets() ;
			forwardPropagation(inputs) ;
			List<Double> outputs = getOutputsAsList() ;
			for (int n = 0; n <= qtdNeuronsInLayer[qtdLayers - 1] - 1; n += 1)
			{
				error += targets.get(n) != 0 ? Math.abs((targets.get(n) - outputs.get(n)) / targets.get(n)) : Math.abs((targets.get(n) - outputs.get(n)) / 1) ;
			}
		}
		return 100 * error / (trainingData.getDataPoints().size() * qtdNeuronsInLayer[qtdLayers - 1])  ;
	}

	protected abstract List<Double> getOutputsAsList() ;

	// protected abstract double calcErrorPerc() ;

	// protected double calcErrorPerc()
	// {
	// 	double error = 0;		
	// 	for (int t = 0; t <= target.length - 1; t += 1)
	// 	{	
	// 		for (int n = 0; n <= qtdNeuronsInLayer[qtdLayers - 1] - 1; n += 1)
	// 		{
	// 			error += target[t][n] != 0 ? Math.abs((target[t][n] - output[t][n]) / target[t][n]) : Math.abs((target[t][n] - output[t][n]) / 1) ;
	// 		}		
	// 	}
	// 	return 100 * error / (target.length * qtdNeuronsInLayer[qtdLayers - 1])  ;
	// }

//	public double calcErrorMethod1()
//	{
//		double error = 0;		
//		for (int inp = 0; inp <= target.length - 1; inp += 1)
//		{	
//			for (int n = 0; n <= Nneurons[Nlayers - 1] - 1; n += 1)
//			{
//				error += 1 / 2.0 * Math.pow(target[inp][n] - output[inp][n], 2);
//			}		
//		}
//		return error;
//	}

	public void run(List<DataPoint> trainingDataPoints)
	{
		switch (mode)
		{
			case train:
				if (qtdIter <= iter) { return ;}
				train(trainingDataPoints) ;
				iter += 1 ;
				
				return ;

			case test:
				test(trainingDataPoints) ;
				
				return ;

			case use:
				use(trainingDataPoints.get(0).getInputs()) ;
				
				return ;
		
			default:
				return ;
		}	
	}

	public void updateResults()
	{
		results.setAvrError(calcAvrErrorPerc()) ;
	}

	public void displayInfoPanel()
	{
		infoPanel.display(biasIsActive, iter, learningRate, results.getAvrError(), mode) ;
	}

	public InfoPanel getInfoPanel() { return infoPanel ;}

}
