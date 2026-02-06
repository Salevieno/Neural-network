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
	protected boolean adaptativeLearningRate ;
    protected boolean biasIsActive ;
	protected boolean biasOnFirstLayer ;
	protected boolean actFunctionOnFirstLayer ;
	protected int[] qtdNeuronsInLayer;
	protected int qtdLayers ;
	protected Results results ;
	protected double trainIterationError ;
	protected double deltaError ;
	protected Mode mode ;
	protected Map<DataPoint, List<Double>> lastOutputsPerDataPoint ;
	protected final ActivationFunction act ; // TODO act function para cada layer

	protected static final double STD_INIT_LEARNING_RATE = 0.5 ;
	protected static final double LEARNING_RATE_MIN = 0.05 ;
	protected static final double LEARNING_RATE_MAX = 1.5 ;
	protected static final boolean DEBUG_MODE = true ;
	protected static final int STD_MAX_ITERATIONS = 100000 ;
	protected static final double MIN_DELTA_ERROR = 0.0000001 ;
	protected static final int[] STD_QTD_NEURONS = new int[] {2, 2, 1, 2, 3} ;
	protected static final double TRAINING_SET_PERCENTAGE = 0.8 ;
	protected static final Data TRAINING_DATA_POINTS = new Data("training_data.json") ;

	public ANN(int[] qtdNeuronsInLayer, ActivationFunction act, boolean adaptativeLearningRate)
	{
        this.iter = 0 ;
        this.qtdIter = STD_MAX_ITERATIONS ;

		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		this.qtdLayers = qtdNeuronsInLayer.length ;
		this.results = new Results() ;
		this.learningRate = adaptativeLearningRate ? LEARNING_RATE_MAX : STD_INIT_LEARNING_RATE ;
		this.adaptativeLearningRate = adaptativeLearningRate ;
		this.trainIterationError = 0 ;
		this.mode = Mode.train ;
		this.lastOutputsPerDataPoint = new HashMap<>() ;
		this.act = act ;
	}

	public abstract void forwardPropagation(List<Double> input) ;
    public abstract void train(List<DataPoint> trainingDataPoints) ;
    public abstract void test(List<DataPoint> testingDataPoints) ;
    public abstract List<Double> use(List<Double> inputs) ;
	protected abstract List<Double> getOutputsAsList() ;

	protected static double updateLRate(double lRate, double error, double min, double max)
	{
		return Math.max(Math.min(lRate - 0.01 * error, max), min) ;
	}

	protected double calcPointError(double target, double output) { return Math.pow(target - output, 2) * 1.0 / 2.0 ;}
	protected double calcPointDError(double target, double output) { return (target - output) ;}

	public Map<DataPoint, List<Double>> getLastOutputsPerDataPoint() { return lastOutputsPerDataPoint ;}

	public void run(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints)
	{
		switch (mode)
		{
			case train:
				if (qtdIter <= iter) { System.out.println("Maximum iterations reached! Training stopped." ); return ;}
				if (1 <= iter && Math.abs(deltaError) <= MIN_DELTA_ERROR) { System.out.println("Minimum delta error reached! Training stopped." ); return ;}
				train(trainingDataPoints) ;
				if (this instanceof ANNMatricialVisual)
				{
					((ANNMatricialVisual) this).updateResults(trainingDataPoints) ;
				}
				
				return ;

			case test:
				test(testingDataPoints) ;
				if (this instanceof ANNMatricialVisual)
				{
					((ANNMatricialVisual) this).updateResults(testingDataPoints) ;
				}
				
				return ;

			case use:
				use(trainingDataPoints.get(0).getInputs()) ;
				
				return ;
		
			default:
				return ;
		}	
	}

	public int[] getQtdNeuronsInLayer() { return qtdNeuronsInLayer ; }
	
	public boolean isBiasActive() { return biasIsActive ; }
	public void activateBiases() { this.biasIsActive = true ;}
	public void deactivateBiases() { this.biasIsActive = false ;}
	public boolean isBiasOnFirstLayer() { return biasOnFirstLayer ; }
	public void activateBiasOnFirstLayer() { this.biasOnFirstLayer = true ;}
	public void deactivateBiasOnFirstLayer() { this.biasOnFirstLayer = false ;}
	public boolean isActFunctionOnFirstLayer() { return actFunctionOnFirstLayer ; }
	public void activateActFunctionOnFirstLayer() { this.actFunctionOnFirstLayer = true ;}
	public void deactivateActFunctionOnFirstLayer() { this.actFunctionOnFirstLayer = false ;}

	public void setMode(Mode mode) { this.mode = mode ; }
}