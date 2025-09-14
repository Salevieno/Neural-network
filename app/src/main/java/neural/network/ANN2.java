package neural.network;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import activationFunctions.ActivationFunction;
import activationFunctions.Sigmoid;

public class ANN2 extends ANN
{

	/*
	 * Equations for backpropagation
	 * 
	 * [Dw] = sum([delta]_o * DO_o) ° [C] for each output o
	 * [C] of layer N = {f'(x0) f'(x1) ... f'(xn)}T * {n0 n1 ... nn}. {f} vector for layer N + 1 and {n} vector for layer N
	 * DO_o = -(target_o - output_o)
	 */

	private List<SimpleMatrix> neuronInputs ;
	private List<SimpleMatrix> neuronOutputs ;
	private List<SimpleMatrix> weights ;
	private List<SimpleMatrix> dWeights ;
	private List<SimpleMatrix> biases ;
	private List<SimpleMatrix> dBiases ;
	private List<SimpleMatrix> deltaMatrices ;

	private static final Point initialPanelPos = new Point(40, 300) ;
	private static final int[] STD_QTD_NEURONS = new int[] {0, 2, 2, 0} ;
	private static final double STD_INIT_LEARNING_RATE = 1.0 ;

    public ANN2(int[] qtdNeuronsInLayer, boolean randomInitialWeights, boolean randomInitialBiases)
    {
		super(initialPanelPos, qtdNeuronsInLayer, new Sigmoid()) ;
		this.neuronInputs = new ArrayList<>() ;
		this.neuronOutputs = new ArrayList<>() ;
		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		this.qtdNeuronsInLayer[0] = trainingData.getDataPoints().get(0).getInputs().size() ;
		this.qtdNeuronsInLayer[qtdNeuronsInLayer.length - 1] = trainingData.getDataPoints().get(0).getTargets().size() ;
		for (int neuron : qtdNeuronsInLayer)
		{
			this.neuronInputs.add(new SimpleMatrix(neuron, 1)) ;
			this.neuronOutputs.add(new SimpleMatrix(neuron, 1)) ;
		}
        this.biasIsActive = false ;
		this.learningRate = STD_INIT_LEARNING_RATE ;
		this.weights = initWeights(qtdLayers, randomInitialWeights) ;
		this.dWeights = initMatricesWithZero(qtdLayers) ;
		this.biases = initBiases(qtdLayers, randomInitialBiases) ;
		this.dBiases = initMatricesWithZero(qtdLayers) ;
		this.deltaMatrices = initMatricesWithZero(qtdLayers) ;
    }

    public ANN2(boolean randomInitialWeights, boolean randomInitialBiases)
    {
        this(STD_QTD_NEURONS, randomInitialWeights, randomInitialBiases) ;
    }

    public ANN2()
    {
        this(true, true) ;
    }

	// private static double[] extractOutputs(List<SimpleMatrix> neurons)
	// {
	// 	SimpleMatrix lastLayer = neurons.get(neurons.size() - 1);
	// 	return lastLayer.getDDRM().getData();
	// }

	private List<SimpleMatrix> initWeights(int Nlayers, boolean randomInitialWeights)
	{
		List<SimpleMatrix> weights = new ArrayList<>() ;

		double startValue = 0.1;
		for (int layer = 0; layer <= Nlayers - 2; layer += 1)
		{
			int qtdNeuronsCurrentLayer = qtdNeuronsInLayer[layer] ;
			int qtdNeuronsNextLayer = qtdNeuronsInLayer[layer + 1] ;

			weights.add(SimpleMatrix.random_DDRM(qtdNeuronsNextLayer, qtdNeuronsCurrentLayer, -0.5, 0.5, new java.util.Random())) ;

			if (!randomInitialWeights)
			{
				SimpleMatrix weightMatrix = weights.get(weights.size() - 1);
				for (int row = 0 ; row <= weightMatrix.getNumRows() - 1 ; row += 1)
				{
					for (int col = 0 ; col <= weightMatrix.getNumCols() - 1 ; col += 1)
					{
						weightMatrix.set(row, col, startValue) ;
						startValue += 0.1 ;
					}
				}
			}
		}

		return weights ;
	}

	private List<SimpleMatrix> initMatricesWithZero(int Nlayers)
	{
		List<SimpleMatrix> deltaMatrices = new ArrayList<>() ;

		for (int i = 0; i <= Nlayers - 2; i += 1)
		{
			deltaMatrices.add(new SimpleMatrix(qtdNeuronsInLayer[i], qtdNeuronsInLayer[i + 1])) ;
		}

		return deltaMatrices ;
	}

	private List<SimpleMatrix> initBiases(int Nlayers, boolean randomInitialBiases)
	{
		List<SimpleMatrix> biases = new ArrayList<>() ;

		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			int qtdNeuronsCurrentLayer = qtdNeuronsInLayer[layer] ;

			biases.add(SimpleMatrix.random_DDRM(qtdNeuronsCurrentLayer, 1, -0.5, 0.5, new java.util.Random())) ;

			if (!randomInitialBiases)
			{
				biases.get(biases.size() - 1).fill(0.5);
			}
		}

		return biases ;
	}

	public void forwardPropagation(List<Double> input)
	{
		// initNeurons() ;
		neuronInputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().mapToDouble(Double::doubleValue).toArray())) ;
		neuronOutputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().mapToDouble(Double::doubleValue).toArray())) ;
		for (int layer = 1; layer <= qtdLayers - 1; layer += 1)
		{
			SimpleMatrix neuronsPrevLayer = neuronOutputs.get(layer - 1) ;
			SimpleMatrix newNeuronInputs = weights.get(layer - 1).mult(neuronsPrevLayer) ;

			if (biasIsActive)
			{
				newNeuronInputs.plus(biases.get(layer - 1)) ;
			}

			neuronInputs.set(layer, newNeuronInputs.copy()) ;

			for (int i = 0 ; i <= newNeuronInputs.getNumElements() - 1 ; i += 1)
			{
				newNeuronInputs.set(i, act.f(newNeuronInputs.get(i))) ;
			}

			neuronOutputs.set(layer, newNeuronInputs) ;			
		}
	}

	public void backPropagation(List<DataPoint> trainingData)
	{
		for (DataPoint dataPoint : trainingData)
		{
			backPropagationIteration(dataPoint) ;
		}
	}

	protected void backPropagationIteration(DataPoint dataPoint)
	{
		List<SimpleMatrix> dWeights = new ArrayList<>() ;
		for (int layer = qtdNeuronsInLayer.length - 1 ; 1 <= layer ; layer += -1)
		{
			// SimpleMatrix deltaMatrix = calcDeltasToLayer(layer, 0) ;
			updateDeltaMatrix(layer - 1, 0);
			// deltaMatrices.set(layer - 1, deltaMatrix) ;
			SimpleMatrix dWeight = calcDWeighMatrix(deltaMatrices.get(layer - 1), dataPoint, layer, 0) ;
			dWeights.add(0, dWeight); // Add to the beginning of the list
		}
		updateWeights(dWeights) ;
	}

	private void updateWeights(List<SimpleMatrix> dWeights)
	{
		for (int i = 0 ; i <= dWeights.size() - 1 ; i+= 1)
		{
			weights.set(i, weights.get(i).plus(dWeights.get(i).scale(-learningRate)));
		}
	}

    public void train(List<DataPoint> trainingData)
    {
        for (int in = 0; in <= trainingData.size() - 1; in += 1)
        {
            forwardPropagation(trainingData.get(in).getInputs()) ;
            backPropagation(trainingData) ;
            // output[in] = getOutputs(in, target) ;
        }

        // error = calcAvrError(output, target) ;
        // if (AdaptativeLrate)
        // {
        //     updateLRate(error) ;
        // }
    }

	public void test(List<DataPoint> trainingDataPoints)
	{

	}

	public List<Double> use(List<Double> inputs)
	{
		forwardPropagation(inputs) ;

		SimpleMatrix outputs = getOutputs() ;

		List<Double> outputList = new ArrayList<>();
		for (int i = 0; i <= outputs.getNumRows() - 1; i += 1)
		{
			outputList.add(outputs.get(i, 0)) ;
		}

		return outputList;
	}

	protected SimpleMatrix getOutputs() { return neuronOutputs.getLast() ;}

	public SimpleMatrix calcDeltasToLayer(int layer, int outputID)
	{
		SimpleMatrix deltaMatrix = new SimpleMatrix(qtdNeuronsInLayer[layer], qtdNeuronsInLayer[layer - 1]) ;

		// last layer
		if (layer == qtdLayers - 1)
		{
			SimpleMatrix rowVector = new SimpleMatrix(1, qtdNeuronsInLayer[layer - 1]);
			rowVector.fill(1.0);

			// Insert a row vector filled with 1s into the specific row of index outputID
			deltaMatrix.insertIntoThis(outputID, 0, rowVector);
			// deltaMatrices.set(layer - 1, deltaMatrix) ;

			return deltaMatrix ;
		}
		
		// before last layer
		if (layer == qtdLayers - 2)
		{
			SimpleMatrix rowVector = weights.get(layer).extractVector(true, outputID).transpose();

			for (int i = 0 ; i <= qtdNeuronsInLayer[layer] - 1 ; i += 1)
			{
				// Create a row vector from the weights of neuron `outputID`
				// SimpleMatrix rowVector = weights.get(layer).extractVector(true, outputID);
				deltaMatrix.insertIntoThis(0, i, rowVector);
			}

			// deltaMatrices.set(layer - 1, deltaMatrix) ;
			return deltaMatrix ;
		}

		// layers before that
		for (int i = 0 ; i <= qtdNeuronsInLayer[layer] - 1 ; i += 1)
		{
			SimpleMatrix rowVector = new SimpleMatrix(1, qtdNeuronsInLayer[layer - 1]) ;
			for (int j = 0 ; j <= qtdNeuronsInLayer[layer - 1] - 1 ; j += 1)
			{
				double delta = calcDelta(layer, outputID, i) ;
				rowVector.set(0, j, delta) ;
			}
			deltaMatrix.insertIntoThis(i, 0, rowVector) ;
		}

		// deltaMatrices.set(layer - 1, deltaMatrix) ;
		return deltaMatrix ;

	}

	public void updateDeltaMatrix(int layer, int outputID)
	{
		SimpleMatrix deltaMatrix = calcDeltasToLayer(layer + 1, 0) ;
		deltaMatrices.set(layer, deltaMatrix) ;
	}

	public SimpleMatrix calcDWeighMatrix(SimpleMatrix deltaMatrix, DataPoint trainingDataPoint, int layer, int outputID)
	{
		double DO = calcOutputErrorPropagatedToLastLayer(trainingDataPoint.getTargets().get(outputID), getOutputs().get(outputID), neuronInputs.get(qtdLayers - 1).get(outputID)) ;
		
		// last layer
		// doesn't multiply by the neurons of the last layer because that's embedded in DO to reduce one repeated multiplication
		if (layer == qtdLayers - 1)
		{
			SimpleMatrix neuronsPrevLayer = neuronOutputs.get(layer - 1);
			for (int i = 0 ; i <= deltaMatrix.getNumCols() - 1 ; i += 1)
			{
				double neuronValue = neuronsPrevLayer.get(i) ;
				SimpleMatrix columnVector = deltaMatrix.extractVector(false, i).scale(neuronValue) ;
				deltaMatrix.insertIntoThis(0, i, columnVector) ;
			}

			SimpleMatrix dWMatrix = deltaMatrix.scale(DO) ;
			return dWMatrix;
		}

		SimpleMatrix neuronProduct = neuronOutputs.get(layer).mult(neuronOutputs.get(layer - 1).transpose()) ;
		SimpleMatrix dWMatrix = deltaMatrix.elementMult(neuronProduct).scale(DO) ;

		return dWMatrix ;
	}

	public double calcDelta(List<SimpleMatrix> deltaMatrices, int layer, int outputID, int col)
	{
		double delta = 0 ; 
		for (int i = 0 ; i <= qtdNeuronsInLayer[layer - 1] - 1 ; i += 1)
		{
			delta += deltaMatrices.get(layer).get(i, col) * neuronOutputs.get(layer + 1).get(i, 0) * weights.get(layer).get(i, col) ;
		}

		return delta ;
	}

	public double calcDelta(int layer, int outputID, int col)
	{
		double delta = 0 ; 
		for (int i = 0 ; i <= qtdNeuronsInLayer[layer - 1] - 1 ; i += 1)
		{
			delta += deltaMatrices.get(layer).get(i, col) * neuronOutputs.get(layer + 1).get(i, 0) * weights.get(layer).get(i, col) ;
		}

		return delta ;
	}

	public double calcOutputError(DataPoint dataPoint, int outputID) { return calcError(dataPoint.getTargets().get(outputID), getOutputs().get(outputID)) ;}

	private double calcDataPointError(DataPoint dataPoint)
	{
		double error = 0 ;

		for (int i = 0 ; i <= getOutputs().getNumCols() - 1 ; i += 1)
		{
			error += calcOutputError(dataPoint, i) ;
		}

		return error ;
	}

	public double calcTotalError(List<DataPoint> trainingData) { return trainingData.stream().map(dp -> calcDataPointError(dp)).mapToDouble(Double::valueOf).sum() ;}



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

		annPanel.display(qtdNeuronsInLayer, trainingData.getDataPoints().get(0).getInputs(), trainingData.getNormalizedDataPoints().get(0).getTargets(), neuronsAsDoubleArray, weightsAsDoubleArray, maxWeight(weightsAsDoubleArray)) ; 
	
	}
	public List<Double> getOutputsAsList() { return Arrays.stream(getOutputs().getDDRM().getData()).boxed().toList() ;}
	public List<SimpleMatrix> getNeuronInputs() { return neuronInputs ;}
	public List<SimpleMatrix> getNeuronOutputs() { return neuronOutputs ;}
	public List<SimpleMatrix> getWeights() { return weights ;}
	public List<SimpleMatrix> getdWeights() { return dWeights ;}
	public List<SimpleMatrix> getBiases() { return biases ;}
	public List<SimpleMatrix> getdBiases() { return dBiases ;}
	public List<SimpleMatrix> getDeltaMatrices() { return deltaMatrices ;}
	public ActivationFunction getAct() { return act ;}

}
