package neural.network;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import activationFunctions.ActivationFunction;
import charts.Chart;
import charts.ChartType;
import charts.Dataset;
import draw.Draw;
import graphics.Align;
import main.Palette;
import utilities.Util;

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
	protected Map<DataPoint, List<Double>> lastOutputsPerDataPoint ;
	protected final ActivationFunction act ;

	protected InfoPanel infoPanel ;
	protected ANNPanel annPanel ;
	protected Dataset trainResultsDataset = new Dataset() ;
	protected final Chart trainResultsGraph = new Chart(ChartType.radar, new Point(705, 435), Align.center, "Training results", 100) ;
	protected Dataset errorDataset = new Dataset() ;
	protected final Chart errorChart = new Chart(ChartType.line, new Point(915, 435), Align.center, "Error", 100) ;

	protected static final Data trainingData = new Data("input.json") ;
	protected static final boolean debugMode = true ;
	protected static final int STD_MAX_ITERATIONS = 100000 ;

	public ANN(Point topLeftPos, int[] qtdNeuronsInLayer, ActivationFunction act)
	{
        this.iter = 0 ;
        this.qtdIter = STD_MAX_ITERATIONS ;
		this.infoPanel = new InfoPanel(new Point(topLeftPos.x, topLeftPos.y)) ;
		this.annPanel = new ANNPanel(new Point(topLeftPos.x + 10 + infoPanel.size.width, topLeftPos.y)) ;
		
		trainResultsDataset = new Dataset();
		trainResultsGraph.addDataset(trainResultsDataset);
		trainResultsGraph.setSize(150) ;
		trainResultsGraph.setPos(Util.translate(topLeftPos, 730, 97));
		trainResultsGraph.setGridColor(Palette.black) ;
		trainResultsGraph.setDataSetColor(List.of(Palette.blue)) ;
		trainResultsGraph.setDataSetContourColor(List.of(Palette.cyan)) ;

		errorDataset= new Dataset() ;
		errorChart.addDataset(errorDataset);
		errorChart.setSize(150) ;
		errorChart.setPos(Util.translate(topLeftPos, 730 + 205, 97));
		errorChart.setGridColor(Palette.black) ;
		errorChart.setDataSetColor(List.of(Palette.purple)) ;
		errorChart.setDataSetContourColor(List.of(Palette.cyan)) ;

		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		this.qtdLayers = qtdNeuronsInLayer.length ;
		this.results = new Results() ;
		this.mode = Mode.train ;
		this.lastOutputsPerDataPoint = new HashMap<>() ;
		this.act = act ;
	}

	public abstract void forwardPropagation(List<Double> input) ;
    public abstract void train(List<DataPoint> trainingDataPoints) ;
    public abstract void test(List<DataPoint> trainingDataPoints) ;
    public abstract List<Double> use(List<Double> inputs) ;
	protected abstract List<Double> getOutputsAsList() ;
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
	protected double calcPointError(double target, double output) { return Math.pow(target - output, 2) * 1.0 / 2.0 ;}
	protected double calcPointDError(double target, double output) { return (target - output) ;}
	
	public double calcTotalError(List<DataPoint> trainingDataPoints)
	{
		double error = 0;		
		for (int t = 0; t <= trainingDataPoints.size() - 1; t += 1)
		{	
			List<Double> targets = trainingDataPoints.get(t).getTargets() ;
			List<Double> outputs = getOutputsAsList() ;
			for (int n = 0; n <= qtdNeuronsInLayer[qtdLayers - 1] - 1; n += 1)
			{
				error += calcPointError(targets.get(n), outputs.get(n)) ;
			}		
		}
		return error ;
	}

	public double calcAvrErrorPerc()
	{
		double error = 0;
		for (int i = 0; i <= trainingData.getDataPoints().size() - 1; i += 1)
		{
			List<Double> inputs = trainingData.getDataPoints().get(i).getInputs() ;
			List<Double> targets = trainingData.getDataPoints().get(i).getTargets() ;
			forwardPropagation(inputs) ;
			List<Double> outputs = getOutputsAsList() ;
			for (int n = 0; n <= qtdNeuronsInLayer[qtdLayers - 1] - 1; n += 1)
			{
				error += targets.get(n) != 0 ? Math.abs((targets.get(n) - outputs.get(n)) / targets.get(n)) : Math.abs((targets.get(n) - outputs.get(n)) / 1) ;
			}
		}
		return 100 * error / (trainingData.getDataPoints().size() * qtdNeuronsInLayer[qtdLayers - 1])  ;
	}

	public Map<DataPoint, List<Double>> getLastOutputsPerDataPoint() { return lastOutputsPerDataPoint ;}

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
				updateResults(trainingDataPoints) ;
				
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

	public void updateResults(List<DataPoint> trainingDataPoints)
	{
		if (qtdIter <= iter) { return ;}

		results.setAvrError(calcTotalError(trainingDataPoints)) ;

		trainResultsDataset.setX(trainingDataPoints.stream().map(trainingDataPoints::indexOf).map(d -> (double)d).toList()) ;
		List<Double> errorOutputsToTargets = new ArrayList<>() ;
		for (int i = 0 ; i <= trainingDataPoints.size() - 1 ; i += 1)
		{
			double target = trainingDataPoints.get(i).getTargets().get(0) ;
			double output = getLastOutputsPerDataPoint().get(trainingDataPoints.get(i)).get(0) ;
			errorOutputsToTargets.add(target != 0 ? Math.abs((target - output) / target) : Math.abs((target - output) / 1)) ;
		}

		// trainResultsDataset.setX(IntStream.rangeClosed(1, trainingDataPoints.get(0).getTargets().size())
		// 									.mapToObj(i -> (double) i)
		// 									.toList()) ;
		// List<Double> errorOutputsToTargets = new ArrayList<>() ;
		// for (int i = 0 ; i <= trainingDataPoints.get(0).getTargets().size() - 1 ; i += 1)
		// {
		// 	double target = trainingDataPoints.get(0).getTargets().get(i) ;
		// 	double output = getLastOutputsPerDataPoint().get(trainingDataPoints.get(0)).get(i) ;
		// 	errorOutputsToTargets.add(target != 0 ? Math.abs((target - output) / target) : Math.abs((target - output) / 1)) ;
		// }

		trainResultsDataset.setY(errorOutputsToTargets) ;

		if (errorDataset.getX().isEmpty())
		{
			errorDataset.addPoint(0, results.getAvrError());
			errorChart.setMaxY(results.getAvrError());
		}
		else
		{
			errorDataset.addPoint(errorDataset.getX().getLast() + 1, results.getAvrError());
			errorChart.setMaxY(Math.max(errorChart.getMaxY(), results.getAvrError()));
			errorChart.setMaxX(errorDataset.getX().getLast() + 1);
		}

		iter += 1 ;		
	}

	public void displayInfoPanel()
	{
		infoPanel.display(biasIsActive, iter, learningRate, results.getAvrError(), mode) ;
	}

	public void displayTrainingResultGraph()
	{
		Draw.menu(trainResultsGraph.getPos(), Align.center);
		trainResultsGraph.display(Draw.DP) ;
	}

	public void displayErrorGraph()
	{
		Draw.menu(errorChart.getPos(), Align.center);
		errorChart.display(Draw.DP) ;
	}

	public InfoPanel getInfoPanel() { return infoPanel ;}

	public int[] getQtdNeuronsInLayer() { return qtdNeuronsInLayer ; }

}
