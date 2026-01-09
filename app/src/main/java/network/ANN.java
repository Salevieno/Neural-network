package network;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
	protected double trainIterationError ;
	protected Mode mode ;
	protected Map<DataPoint, List<Double>> lastOutputsPerDataPoint ;
	protected final ActivationFunction act ; // TODO act function para cada layer

	protected static final boolean DEBUG_MODE = true ;
	protected static final int STD_MAX_ITERATIONS = 100000 ;
	protected static final int[] STD_QTD_NEURONS = new int[] {2, 2, 1, 2, 3} ;
	protected static final Data TRAINING_DATA_POINTS = new Data("training_data.json") ;

	public ANN(int[] qtdNeuronsInLayer, ActivationFunction act)
	{
        this.iter = 0 ;
        this.qtdIter = STD_MAX_ITERATIONS ;

		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		this.qtdLayers = qtdNeuronsInLayer.length ;
		this.results = new Results() ;
		this.trainIterationError = 0 ;
		this.mode = Mode.train ;
		this.lastOutputsPerDataPoint = new HashMap<>() ;
		this.act = act ;
	}

	public abstract void forwardPropagation(List<Double> input) ;
    public abstract void train(List<DataPoint> trainingDataPoints) ;
    public abstract void test(List<DataPoint> trainingDataPoints) ;
    public abstract List<Double> use(List<Double> inputs) ;
	protected abstract List<Double> getOutputsAsList() ;

	protected static double updateLRate(double lRate, double error, double min, double max) { return Math.max(Math.min(lRate - 0.01 * (error - 0.6), max), min) ;}

	protected double calcPointError(double target, double output) { return Math.pow(target - output, 2) * 1.0 / 2.0 ;}
	protected double calcPointDError(double target, double output) { return (target - output) ;}

	public Map<DataPoint, List<Double>> getLastOutputsPerDataPoint() { return lastOutputsPerDataPoint ;}

	public void run(List<DataPoint> trainingDataPoints)
	{
		switch (mode)
		{
			case train:
				if (qtdIter <= iter) { return ;}
				train(trainingDataPoints) ;
				if (this instanceof ANNMatricialVisual)
				{
					((ANNMatricialVisual) this).updateResults(trainingDataPoints) ;
				}
				
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

	public int[] getQtdNeuronsInLayer() { return qtdNeuronsInLayer ; }

}