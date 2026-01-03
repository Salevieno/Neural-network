package neural.network;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import activationFunctions.Sigmoid;

public class ANN1 extends ANN
{

	private double[][] neuronvalue;
	private double[][][] weights;
	private double[][][] Dweight;
	private double[][] biases;

	private List<Double[]> output;

	private int[] multvec;
	
	// private boolean adaptativeLRate ;
	
	// private static final double L_RATE_MAX = 0.6 ;
	// private static final double L_RATE_MIN = 0.0005 ;
	private static final Point INITIAL_PANEL_POS = new Point(40, 60) ;
	private static final int[] STD_QTD_NEURONS = new int[] {2, 2, 2, 3} ;
	private static final List<DataPoint> TRAINING_DATA_POINTS = new Data("input.json").getDataPoints() ;
	
	public ANN1(List<DataPoint> trainingData, boolean randomInitialWeights)
	{
		super(INITIAL_PANEL_POS, STD_QTD_NEURONS, new Sigmoid()) ;
		
        this.iter = 0 ;
		this.qtdNeuronsInLayer = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;
		this.neuronvalue = initNeurons(qtdLayers, this.qtdNeuronsInLayer) ;
        this.biasIsActive = false ;
		this.learningRate = 1.0 ;
		this.weights = initWeights(qtdLayers, randomInitialWeights) ;
		this.biases = initBiases(qtdLayers) ;
		this.multvec = calcProdVec(qtdLayers, qtdNeuronsInLayer);
		this.output = new ArrayList<>() ;
	}
	
	public ANN1(boolean randomInitialWeights)
	{
		this(TRAINING_DATA_POINTS, randomInitialWeights) ;
	}

	private double[][] initNeurons(int Nlayers, int[] Nneurons)
	{
		double[][] neuronValues = new double[Nlayers][];
		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			neuronValues[layer] = new double[Nneurons[layer]];
		}
		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			for (int ni = 0; ni <= Nneurons[layer] - 1; ni += 1)
			{
				neuronValues[layer][ni] = 0;
			}
		}

		return neuronValues ;
	}

	private double[][][] initWeights(int Nlayers, boolean randomInitialWeights)
	{
		double[][][] weights ;
		weights = new double[Nlayers - 1][][];

		double startValue = 0.1;

		for (int layer = 0; layer <= Nlayers - 2; layer += 1)
		{
			weights[layer] = new double[qtdNeuronsInLayer[layer + 1]][qtdNeuronsInLayer[layer]];
		}
		for (int layer = 0; layer <= Nlayers - 2; layer += 1)
		{
			for (int ni = 0; ni <= qtdNeuronsInLayer[layer + 1] - 1; ni += 1)
			{
				for (int nf = 0; nf <= qtdNeuronsInLayer[layer] - 1; nf += 1)
				{
					weights[layer][ni][nf] = randomInitialWeights ? 0.5 * (Math.random() - Math.random()) : startValue ;
					startValue += 0.1 ;
				}
			}
		}

		return weights ;
	}

	private double [][] initBiases(int Nlayers)
	{
		double[][] biases ;
		biases = new double[Nlayers][];
		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			biases[layer] = new double[qtdNeuronsInLayer[layer]];
		}
		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			for (int ni = 0; ni <= qtdNeuronsInLayer[layer] - 1; ni += 1)
			{
				biases[layer][ni] = 0.05;
			}
		}

		return biases ;
	}

	public double[][] forwardPropagationWithReturn(List<Double> input)
	{
		neuronvalue = new double[qtdLayers][] ;
		neuronvalue[0] = input.stream().mapToDouble(Double::valueOf).toArray();
		for (int layer = 1; layer <= qtdLayers - 1; layer += 1)
		{
			neuronvalue[layer] = vecMatrixProd(neuronvalue[layer - 1], weights[layer - 1]);
			if (biasIsActive)
			{
				neuronvalue[layer] = vecMatrixProdWithBias(neuronvalue[layer - 1], weights[layer - 1], biases[layer - 1]);
			}
			for (int n = 0; n <= qtdNeuronsInLayer[layer] - 1; n += 1)
			{
				neuronvalue[layer][n] = act.f(neuronvalue[layer][n]);
			}
			
		}
		return neuronvalue;
	}

	public void forwardPropagation(List<Double> input)
	{
		forwardPropagationWithReturn(input) ;
	}


	private static int[] calcProdVec(int Nlayers, int[] Nneurons)
	{
		int[] multvec = new int[Nlayers];
		multvec[Nlayers - 1] = 1;
		for (int i = Nlayers - 1; 1 <= i; i += -1)
		{
			multvec[i - 1] = multvec[i]*Nneurons[i];
		}
		return multvec;
	}
	
	private static double[] vecMatrixProd(double[] vector, double[][] matrix)
	{
		if (vector.length != matrix[0].length)
		{
			System.out.println("Error: Attempted to multiply matrices of different sizes at UtilGeral -> MatrixProd");
			System.out.println("Vector size: " + vector.length + " Matrix size : " + matrix[0].length);
			return null;
		}
		else
		{
			double product[] = new double[matrix.length];
			for (int i = 0; i <= matrix.length - 1; i += 1) 
			{
				for (int j = 0; j <= vector.length - 1; j += 1) 
				{
					product[i] += vector[j] * matrix[i][j];
				}
			}		
			return product;
		}
	}

	private static double[] vecMatrixProdWithBias(double[] vector, double[][] matrix, double[] bias)
	{
		if (vector.length != matrix[0].length)
		{
			System.out.println("Error: Attempted to multiply matrices of different sizes");
			System.out.println("Vector size: " + vector.length + " Matrix size : " + matrix[0].length);
			return null;
		}
		else
		{
			double product[] = new double[matrix.length];
			for (int i = 0; i <= matrix.length - 1; i += 1) 
			{
				for (int j = 0; j <= vector.length - 1; j += 1) 
				{
					if (bias[j] < vector[j])
					{
						product[i] += vector[j] * matrix[i][j];
					}
				}
			}		
			return product;
		}
	}
	
	public double[][][] backPropagation(List<Double> target)
	{
		int Nlayers = qtdNeuronsInLayer.length;
		Dweight = new double[Nlayers - 1][][] ;
		for (int layer = 0; layer <= Nlayers - 2; layer += 1)
		{
			Dweight[layer] = new double[qtdNeuronsInLayer[layer + 1]][qtdNeuronsInLayer[layer]];
		}
		for (int layer = Nlayers - 1; 1 <= layer; layer += -1)			// propagate from targets to input
		{	
			for (int wi = 0; wi <= qtdNeuronsInLayer[layer - 1] - 1; wi += 1)	// weight from neuron i
			{
				for (int wf = 0; wf <= qtdNeuronsInLayer[layer] - 1; wf += 1)	// to neuron f in the next layer
				{
					if (layer < Nlayers - 1)
					{
						double SumD = 0;
						for (int path = 0; path <= multvec[layer] - 1; path += 1)	// For each possible path, create a map
						{
							int[] Map = createMap(path, Nlayers - layer, wi, wf, qtdNeuronsInLayer, multvec);
							Map[Map.length - 1] = wf;
							
							double ProdD = 1;
							for (int MapID = 1; MapID <= Map.length - 1; MapID += 1)
							{
								double Dsig = act.df(neuronvalue[layer + MapID][Map[Map.length - MapID - 1]]);
								double W = weights[layer + MapID - 1][Map[Map.length - MapID - 1]][Map[Map.length - MapID]];	// lWif está registrado como lWfi
								ProdD = ProdD * Dsig * W;
							}
							SumD += ProdD * dEdy(target.get(Map[0]), neuronvalue[Nlayers - 1][Map[0]]);
						}
						double Dsig = act.df(neuronvalue[layer][wf]);
						double N = neuronvalue[layer - 1][wi];
						Dweight[layer - 1][wf][wi] = SumD * Dsig * N;
					}
					else
					{
						double Dsig = act.df(neuronvalue[Nlayers - 1][wf]);
						double N = neuronvalue[layer - 1][wi];
						Dweight[layer - 1][wf][wi] = dEdy(target.get(wf), neuronvalue[Nlayers - 1][wf]) * Dsig * N;
					}
				}	
			}
		}

		UpdateWeights() ;
		return Dweight;
	}
	

	private double dEdy(double target, double neuronvalue)
	{
		return -(target - neuronvalue);
	}
	
	private static int[] createMap(int path, int layer, int wi, int wf, int[] Nneurons, int[] multvec)
	{
		int Nlayers = Nneurons.length;
		int[] Map = new int[layer];
		for (int l = 0; l <= layer - 1; l += 1)
		{
			Map[l] = (path / multvec[Nlayers - l - 1]) % Nneurons[Nlayers - l - 1];
		}
		return Map;
	}
	
	public void train(List<DataPoint> trainingDataPoints)
	{
		for (int in = 0; in <= trainingDataPoints.size() - 1; in += 1)
		{
			forwardPropagationWithReturn(trainingDataPoints.get(in).getInputs()) ;
			backPropagation(trainingDataPoints.get(in).getTargets()) ;
		}
	}

	public void trainIteration(List<DataPoint> trainingDataPoints)
	{
		if (qtdIter <= iter) { return ;}

		/*
			* There is another method, which is recording the weights here and using them
			* in the forward propagation in the loop (the weights do not change during the
			* iteration) However, the current method seems to perform better, converging in
			* less iterations
			*/
//			double prevError = calcAvrError(output, target);

		output = new ArrayList<>() ;
		for (int in = 0; in <= trainingDataPoints.size() - 1; in += 1)
		{
			forwardPropagationWithReturn(trainingDataPoints.get(in).getInputs()) ;
			backPropagation(trainingDataPoints.get(in).getTargets()) ;
			output.add(getOutput().stream().toArray(Double[]::new)) ;
		}

		// error = calcAvrError(output, trainingDataPoints) ;
		// if (AdaptativeLrate)
		// {
		// 	learningRate = updateLRate(learningRate, error, L_RATE_MIN, L_RATE_MAX) ;
		// }
		
		//			derror = Math.abs(error - prevError);
		iter += 1;
	}
	
	// public void saveOutputFile()
	// {
		//			for (int in = 0; in <= input.length - 1; in += 1)
//			{
//				ForwardPropagation(input[in]) ;
//				output[in] = getOutputs(in, target) ;
//			}
//			Utg.SaveTextFile("Error", SaveError);
	// }

	public void runTrainingOnce(List<DataPoint> trainingDataPoints)
	{
		output = new ArrayList<>() ;
		for (int in = 0; in <= trainingDataPoints.size() - 1; in += 1)
		{
			forwardPropagationWithReturn(trainingDataPoints.get(in).getInputs()) ;
			backPropagation(trainingDataPoints.get(in).getTargets()) ;
			output.add(getOutputsAsList().stream().toArray(Double[]::new)) ;
		}

		// error = calcAvrError(output, trainingDataPoints) ;
		// if (AdaptativeLrate)
		// {
		// 	learningRate = updateLRate(learningRate, error, L_RATE_MIN, L_RATE_MAX) ;
		// }
	}
	
	public void test(List<DataPoint> trainingDataPoints)
	{

	}

	public List<Double> use(List<Double> inputs)
	{
		double[][] neuronValues = forwardPropagationWithReturn(inputs) ;
		double[] outputs = neuronValues[neuronValues.length - 1] ;

		return Arrays.stream(outputs).boxed().toList() ;
	}

	private double[][][] UpdateWeights()
	{
		for (int layer = 0; layer <= qtdLayers - 2; layer += 1)
		{
			for (int ni = 0; ni <= qtdNeuronsInLayer[layer] - 1; ni += 1)
			{	
				for (int nf = 0; nf <= qtdNeuronsInLayer[layer + 1] - 1; nf += 1)
				{
					weights[layer][nf][ni] += - learningRate * Dweight[layer][nf][ni];
				}	
			}
		}
		
		return weights;
	}
	
	public double[][][] getWeights() {return weights ;}
	public double[][] getBiases() {return biases ;}
	public List<Double[]> getOutput() {return output ;}
	public List<Double> getOutputsAsList() { return Arrays.stream(neuronvalue[qtdLayers - 1]).boxed().toList() ;}

	private static double maxWeight(double[][][] weight)
	{		
		double MaxWeight = weight[0][0][0];
		for (int i = 0; i <= weight.length - 1; i += 1)
        {
			for (int j = 0; j <= weight[i].length - 1; j += 1)
	        {
				for (int k = 0; k <= weight[i][j].length - 1; k += 1)
		        {
					if (MaxWeight < Math.abs(weight[i][j][k]))
					{
						MaxWeight = Math.abs(weight[i][j][k]);
					}
		        }
	        }
        }
		return MaxWeight ;
	}
	
	public void display()
	{
		annPanel.display(qtdNeuronsInLayer, trainingData.getDataPoints().get(0).getInputs(), trainingData.getNormalizedDataPoints().get(0).getTargets(), neuronvalue, weights, maxWeight(weights)) ;
	}

}
