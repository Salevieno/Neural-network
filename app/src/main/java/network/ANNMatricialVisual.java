package network ;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

import org.ejml.data.DMatrixRMaj;

import charts.Chart;
import charts.ChartType;
import charts.Dataset;
import draw.Draw;
import graphics.Align;
import main.Palette;
import utilities.Util;

public class ANNMatricialVisual extends ANNMatricial
{
	protected InfoPanel infoPanel ;
	protected ANNPanel annPanel ;

	protected static final Point INITIAL_PANEL_POS = new Point(40, 60) ;
	protected Dataset trainResultsDataset = new Dataset() ;
	protected final Chart trainResultsGraph = new Chart(ChartType.radar, new Point(705, 435), Align.center, "Training results", 100) ;
	protected Dataset errorDataset = new Dataset() ;
	protected final Chart errorChart = new Chart(ChartType.line, new Point(915, 435), Align.center, "Error", 100) ;
    
    public ANNMatricialVisual(Point topLeftPos, int[] qtdNeuronsInLayer, boolean randomizeInitialWeights, boolean randomizeInitialBiases)
    {
        super(qtdNeuronsInLayer, randomizeInitialWeights, randomizeInitialBiases) ;
        
		this.infoPanel = new InfoPanel(new Point(topLeftPos.x, topLeftPos.y)) ;
		this.annPanel = new ANNPanel(new Point(topLeftPos.x + 10 + infoPanel.size.width, topLeftPos.y)) ;

		errorDataset= new Dataset() ;
		errorChart.addDataset(errorDataset);
		errorChart.setSize(150) ;
		errorChart.setPos(Util.translate(INITIAL_PANEL_POS, 730 + 205, 97));
		errorChart.setGridColor(Palette.black) ;
		errorChart.setDataSetColor(List.of(Palette.purple)) ;
		errorChart.setDataSetContourColor(List.of(Palette.cyan)) ;
		
		trainResultsDataset = new Dataset();
		trainResultsGraph.addDataset(trainResultsDataset);
		trainResultsGraph.setSize(150) ;
		trainResultsGraph.setPos(Util.translate(INITIAL_PANEL_POS, 730, 97));
		trainResultsGraph.setGridColor(Palette.black) ;
		trainResultsGraph.setDataSetColor(List.of(Palette.blue)) ;
		trainResultsGraph.setDataSetContourColor(List.of(Palette.cyan)) ;
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

	private static double maxWeight(double[][][] weights)
	{		
		double MaxWeight = weights[0][0][0];
		for (int i = 0; i <= weights.length - 1; i += 1)
        {
			for (int j = 0; j <= weights[i].length - 1; j += 1)
	        {
				for (int k = 0; k <= weights[i][j].length - 1; k += 1)
		        {
					if (MaxWeight < Math.abs(weights[i][j][k]))
					{
						MaxWeight = Math.abs(weights[i][j][k]);
					}
		        }
	        }
        }
		return MaxWeight ;
	}
    
	public InfoPanel getInfoPanel() { return infoPanel ;}

    
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

	public void displayInfoPanel()
	{
		infoPanel.display(biasIsActive, iter, learningRate, results.getAvrError(), mode) ;
	}

	public void display()
	{

        double[][] neuronsAsDoubleArray = new double[qtdLayers][] ;
        double[][][] weightsAsDoubleArray = new double[qtdLayers - 1][][] ;

        for (int i = 0 ; i <= qtdLayers - 1 ; i += 1)
        {
            neuronsAsDoubleArray[i] = neuronOutputs.get(i).getDDRM().getData() ;
        }

        for (int i = 0; i <= qtdLayers - 2; i += 1)
        {
            DMatrixRMaj dMatrix = weights.get(i).getDDRM() ;
            double[][] weightArray = new double[dMatrix.numRows][dMatrix.numCols] ;

            for (int row = 0 ; row <= dMatrix.numRows - 1 ; row += 1)
            {
                for (int col = 0 ; col <= dMatrix.numCols - 1 ; col += 1)
                {
                    weightArray[row][col] = dMatrix.get(row, col) ;
                }
            }

            weightsAsDoubleArray[i] = weightArray ;
        }

		annPanel.display(qtdNeuronsInLayer, TRAINING_DATA_POINTS.getDataPoints().get(0).getInputs(), TRAINING_DATA_POINTS.getDataPoints().get(0).getTargets(), neuronsAsDoubleArray, weightsAsDoubleArray, maxWeight(weightsAsDoubleArray)) ; 
	}
}