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
	protected final InfoPanel infoPanel ;
	protected final ANNPanel annPanel ;
	protected final Chart errorDatapointChart ;
	protected final Chart errorTotalChart ;
    
    public ANNMatricialVisual(Point topLeftPos, int[] qtdNeuronsInLayer, boolean randomizeInitialWeights, boolean randomizeInitialBiases, boolean adaptativeLearningRate)
    {
        super(qtdNeuronsInLayer, randomizeInitialWeights, randomizeInitialBiases, adaptativeLearningRate) ;
        
		this.infoPanel = new InfoPanel(new Point(topLeftPos.x, topLeftPos.y)) ;
		this.annPanel = new ANNPanel(new Point(topLeftPos.x + 10 + infoPanel.size.width, topLeftPos.y)) ;

		errorTotalChart = new Chart(ChartType.line, new Point(915, 435), Align.center, "Error (total)", 150) ;
		errorTotalChart.addDataset(new Dataset());
		errorTotalChart.addDataset(new Dataset());
		errorTotalChart.setPos(Util.translate(topLeftPos, 730 + 205, 97));
		errorTotalChart.setDataSetColor(List.of(Palette.purple, Palette.green)) ;
		
		errorDatapointChart = new Chart(ChartType.radar, new Point(705, 435), Align.center, "Error", 150) ;
		errorDatapointChart.addDataset(new Dataset());
		errorDatapointChart.addDataset(new Dataset());
		errorDatapointChart.setPos(Util.translate(topLeftPos, 730, 97));
		errorDatapointChart.setDataSetColor(List.of(Palette.blue, Palette.green)) ;
    }

	public void updateResults(List<DataPoint> dataPoints)
	{
		int datasetIndex = mode.equals(Mode.train) ? 0 : 1 ;

		errorDatapointChart.getData().get(datasetIndex).setX(dataPoints.stream().map(dataPoints::indexOf).map(d -> (double)d).toList()) ;
		List<Double> errorOutputsToTargets = new ArrayList<>() ;
		for (DataPoint dp : dataPoints)
		{
			double target = dp.getTargets().get(0) ;
			double output = getLastOutputsPerDataPoint().get(dp).get(0) ;
			double outputError = (target + output) != 0 ? Math.abs((target - output) / (target + output)) : Math.abs((target - output) / 1) ;
			errorOutputsToTargets.add(outputError) ;
		}
		errorDatapointChart.getData().get(datasetIndex).setY(errorOutputsToTargets) ;		

		if (errorTotalChart.getData().get(datasetIndex).getX().isEmpty())
		{
			errorTotalChart.getData().get(datasetIndex).addPoint(0, results.getAvrError());
			errorTotalChart.setMaxY(results.getAvrError());
		}
		else
		{
			errorTotalChart.getData().get(datasetIndex).addPoint(errorTotalChart.getData().get(datasetIndex).getX().getLast() + 1, results.getAvrError());
			errorTotalChart.setMaxY(Math.max(errorTotalChart.getMaxY(), results.getAvrError()));
			errorTotalChart.setMaxX(errorTotalChart.getData().get(datasetIndex).getX().getLast() + 1);
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
		Draw.menu(errorDatapointChart.getPos(), Align.center);
		errorDatapointChart.display(Draw.DP) ;
	}

	public void displayErrorGraph()
	{
		Draw.menu(errorTotalChart.getPos(), Align.center);
		errorTotalChart.display(Draw.DP) ;
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