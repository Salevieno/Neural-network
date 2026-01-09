package network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

import activationFunctions.ActivationFunction;
import activationFunctions.Sigmoid;

public class ANNMatricial extends ANN
{
	private List<SimpleMatrix> neuronInputs ;
	protected List<SimpleMatrix> neuronOutputs ;
	protected List<SimpleMatrix> weights ;
	private List<SimpleMatrix> dWeights ;
	private List<SimpleMatrix> biases ;
	private List<SimpleMatrix> dBiases ;
	private List<List<SimpleMatrix>> deltaMatrices ;

	private static final double STD_INIT_LEARNING_RATE = 0.5 ;

    public ANNMatricial(int[] qtdNeuronsInLayer, boolean randomizeInitialWeights, boolean randomizeInitialBiases)
    {
		super(qtdNeuronsInLayer, new Sigmoid()) ;
		this.neuronInputs = new ArrayList<>() ;
		this.neuronOutputs = new ArrayList<>() ;
		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		for (int neuron : qtdNeuronsInLayer)
		{
			this.neuronInputs.add(new SimpleMatrix(neuron, 1)) ;
			this.neuronOutputs.add(new SimpleMatrix(neuron, 1)) ;
		}
        this.biasIsActive = false ;
		this.learningRate = STD_INIT_LEARNING_RATE ;
		this.weights = initWeights(qtdLayers, randomizeInitialWeights) ;
		this.dWeights = initMatricesWithZero(qtdLayers) ;
		this.biases = initBiases(qtdLayers, randomizeInitialBiases) ;
		this.dBiases = initMatricesWithZero(qtdLayers) ;
		this.deltaMatrices = new ArrayList<>() ;
    }

    public ANNMatricial(boolean randomizeInitialWeights, boolean randomizeInitialBiases)
    {
        this(STD_QTD_NEURONS, randomizeInitialWeights, randomizeInitialBiases) ;
    }

    public ANNMatricial()
    {
        this(true, true) ;
    }

	private List<SimpleMatrix> initWeights(int Nlayers, boolean randomInitialWeights)
	{
		List<SimpleMatrix> weights = new ArrayList<>() ;

		double startValue = 1.0 ;
		final double inc = 0.0 ;
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
						startValue += inc ;
					}
				}
				// startValue += inc ;
			}
		}

		return weights ;
	}

	private List<SimpleMatrix> initMatricesWithZero(int Nlayers)
	{
		List<SimpleMatrix> deltaMatrices = new ArrayList<>() ;

		for (int i = 0; i <= Nlayers - 2; i += 1)
		{
			deltaMatrices.add(new SimpleMatrix(qtdNeuronsInLayer[i + 1], qtdNeuronsInLayer[i])) ;
		}
		return deltaMatrices ;
	}

	private List<SimpleMatrix> initBiases(int Nlayers, boolean randomInitialBiases)
	{
		List<SimpleMatrix> biases = new ArrayList<>() ;

		double startValue = 0.35 ;
		final double inc = 0.25 ;
		for (int layer = 0; layer <= Nlayers - 2; layer += 1)
		{
			int qtdNeuronsCurrentLayer = qtdNeuronsInLayer[layer + 1] ;

			biases.add(SimpleMatrix.random_DDRM(qtdNeuronsCurrentLayer, 1, -0.5, 0.5, new java.util.Random())) ;

			if (!randomInitialBiases)
			{
				biases.get(biases.size() - 1).fill(startValue);
				startValue += inc ;
			}
		}

		return biases ;
	}

	public void forwardPropagation(List<Double> input)
	{
		neuronInputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().mapToDouble(Double::doubleValue).toArray())) ;
		neuronOutputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().map(d -> d).mapToDouble(Double::valueOf).toArray())) ;

		for (int layer = 1; layer <= qtdLayers - 1; layer += 1)
		{
			SimpleMatrix neuronsPrevLayer = neuronOutputs.get(layer - 1) ;
			SimpleMatrix newNeuronInputs = weights.get(layer - 1).mult(neuronsPrevLayer) ;

			if (biasIsActive)
			{
				newNeuronInputs = newNeuronInputs.plus(biases.get(layer - 1)) ;
			}

			neuronInputs.set(layer, newNeuronInputs.copy()) ;

			SimpleMatrix newNeuronOutputs = weights.get(layer - 1).mult(neuronsPrevLayer) ;
			for (int i = 0 ; i <= newNeuronOutputs.getNumElements() - 1 ; i += 1)
			{
				newNeuronOutputs.set(i, act.f(newNeuronInputs.get(i))) ;
			}

			neuronOutputs.set(layer, newNeuronOutputs.copy()) ;
		}
	}

	public void backPropagationIteration(DataPoint dataPoint)
	{
		deltaMatrices = new ArrayList<>(Collections.nCopies(qtdLayers - 1, new ArrayList<>())) ;

		for (int layer = qtdLayers - 1 ; 1 <= layer ; layer += -1)
		{
			SimpleMatrix cMatrix = calcCMatrix(layer - 1) ;
			
			deltaMatrices.set(layer - 1, new ArrayList<>()) ;
			for (int outputID = 0 ; outputID <= qtdNeuronsInLayer[qtdLayers - 1] - 1 ; outputID += 1)
			{
				SimpleMatrix deltaMatrix = calcDeltasToLayer(layer - 1, outputID) ;
				deltaMatrices.get(layer - 1).add(deltaMatrix);
			}

			SimpleMatrix dWeightsMatrix = calcDWeights(layer, cMatrix, deltaMatrices.get(layer - 1), dataPoint.getTargets()) ;
			dWeights.set(layer - 1, dWeightsMatrix) ;
		}
		updateWeights(dWeights) ;
	}

	private SimpleMatrix calcDWeights(int layer, SimpleMatrix cMatrix, List<SimpleMatrix> deltaMatrices, List<Double> targets)
	{
		SimpleMatrix dWeights = new SimpleMatrix(qtdNeuronsInLayer[layer], qtdNeuronsInLayer[layer - 1]);
		for (int outputID = 0 ; outputID <= qtdNeuronsInLayer[qtdLayers - 1] - 1 ; outputID += 1)
		{
			double DO = calcPointDError(targets.get(outputID), neuronOutputs.get(qtdLayers - 1).get(outputID)) ;
			dWeights = dWeights.plus(deltaMatrices.get(outputID).scale(DO));
		}
		dWeights = dWeights.elementMult(cMatrix) ;
		dWeights = dWeights.scale(learningRate);
		return dWeights ;
	}

	private void updateWeights(List<SimpleMatrix> dWeights)
	{
		for (int i = 0 ; i <= weights.size() - 1 ; i+= 1)
		{
			weights.set(i, weights.get(i).plus(dWeights.get(i)));
		}
	}

    public void train(List<DataPoint> trainingData) // TODO method name = run train iteration
    {
		deltaError = trainIterationError ;
		trainIterationError = 0 ;
        for (DataPoint dataPoint : trainingData)
		{
			forwardPropagation(dataPoint.getInputs()) ;
			backPropagationIteration(dataPoint) ;
			lastOutputsPerDataPoint.put(dataPoint, getOutputsAsList()) ;
			trainIterationError += calcDataPointError(dataPoint) ;
		}
		deltaError = trainIterationError - deltaError ;
		results.setAvrError(trainIterationError);
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

	private SimpleMatrix calcCMatrix(int layer)
	{
		// [C] of layer N = {f'(x0) f'(x1) ... f'(xn)}^T * {n0 n1 ... nn}. {f} vector for layer N + 1 and {n} vector for layer N
		SimpleMatrix cMatrix = neuronOutputs.get(layer).mult(derivativeMatrix(neuronOutputs.get(layer + 1)).transpose()).transpose() ;
		return cMatrix ;
	}

	private SimpleMatrix calcDeltasToLayer(int layer, int outputID)
	{
		SimpleMatrix deltaMatrix = new SimpleMatrix(qtdNeuronsInLayer[layer + 1], qtdNeuronsInLayer[layer]) ;

		// last layer
		if (layer == qtdLayers - 2)
		{
			SimpleMatrix rowVector = new SimpleMatrix(1, qtdNeuronsInLayer[layer]);
			rowVector.fill(1.0);

			// Insert a row vector filled with 1s into the specific row of index outputID
			deltaMatrix.insertIntoThis(outputID, 0, rowVector);

			return deltaMatrix ;
		}
		
		// before last layer
		if (layer == qtdLayers - 3)
		{
			SimpleMatrix derivativeMatrix = derivativeMatrix(neuronOutputs.get(layer + 2)) ;
			SimpleMatrix weightRow = weights.get(layer + 1).extractVector(true, outputID) ;
			SimpleMatrix deltaMatrixCol = weightRow.transpose().scale(derivativeMatrix.get(outputID)) ;
			for (int col = 0 ; col <= deltaMatrix.getNumCols() - 1 ; col += 1)
			{
				deltaMatrix.insertIntoThis(0, col, deltaMatrixCol);
			}

			return deltaMatrix ;
		}

		// layers before that
		SimpleMatrix dNeuronVector = derivativeMatrix(neuronOutputs.get(layer + 2)) ;
		SimpleMatrix deltaMatrixCol = dNeuronVector.transpose().mult(deltaMatrices.get(layer + 1).get(outputID).elementMult(weights.get(layer + 1))) ;
		for (int row = 0 ; row <= deltaMatrix.getNumRows() - 1 ; row += 1)
		{
			deltaMatrix.insertIntoThis(row, 0, SimpleMatrix.filled(1, deltaMatrix.getNumCols(), deltaMatrixCol.get(0, row)));
		}

		return deltaMatrix ;

	}

	public double calcOutputError(DataPoint dataPoint, int outputID) { return calcPointError(dataPoint.getTargets().get(outputID), getOutputs().get(outputID)) ;}

	private double calcDataPointError(DataPoint dataPoint)
	{
		double error = 0 ;

		for (int i = 0 ; i <= getOutputs().getNumRows() - 1 ; i += 1)
		{
			error += calcOutputError(dataPoint, i) ;
		}

		return error ;
	}

	public double calcTotalError(List<DataPoint> trainingData)
	{
		double totalError = 0 ;
        for (DataPoint dataPoint : trainingData)
		{
			forwardPropagation(dataPoint.getInputs()) ;
			totalError += calcDataPointError(dataPoint) ;
		}
		return totalError ;
	}

	private SimpleMatrix derivativeMatrix(SimpleMatrix matrix) {
		// TODO passar isso para a act
		return matrix.elementMult(matrix.scale(-1).plus(1));
	}

	public List<Double> getOutputsAsList()
	{
		return Arrays.stream(getOutputs().getDDRM().getData()).boxed().toList() ;
	}

	public List<SimpleMatrix> getNeuronInputs() { return neuronInputs ;}
	public List<SimpleMatrix> getNeuronOutputs() { return neuronOutputs ;}
	public List<SimpleMatrix> getWeights() { return weights ;}
	public void setWeights(List<SimpleMatrix> weights) { this.weights = weights ;}
	public List<SimpleMatrix> getdWeights() { return dWeights ;}
	public List<SimpleMatrix> getBiases() { return biases ;}
	public void setBiases(List<SimpleMatrix> biases) { this.biases = biases ;}
	public List<SimpleMatrix> getdBiases() { return dBiases ;}
	public List<List<SimpleMatrix>> getDeltaMatrices() { return deltaMatrices ;}
	public ActivationFunction getAct() { return act ;}
	public double getTrainIterationError() { return trainIterationError ;}
	public double getDeltaError() { return deltaError ;}

	public void printState()
	{
		System.out.println("\n=== ANN2 STATE ===\n") ;
		System.out.println("Neuron Inputs:") ;
		for (SimpleMatrix matrix : neuronInputs)
		{
			matrix.print() ;
		}
		System.out.println("Neuron Outputs:") ;
		for (SimpleMatrix matrix : neuronOutputs)
		{
			matrix.print() ;
		}
		System.out.println("Weights:") ;
		for (SimpleMatrix matrix : weights)
		{
			matrix.print() ;
		}
		System.out.println("dWeights:") ;
		for (SimpleMatrix matrix : dWeights)
		{
			matrix.print() ;
		}
		System.out.println("Biases:") ;
		for (SimpleMatrix matrix : biases)
		{
			matrix.print() ;
		}
		System.out.println("dBiases:") ;
		for (SimpleMatrix matrix : dBiases)
		{
			matrix.print() ;
		}
		System.out.println("Delta Matrices:") ;
		for (List<SimpleMatrix> list : deltaMatrices)
		{
			for (SimpleMatrix matrix : list)
			{
				matrix.print() ;
			}
		}
		System.out.println("Derivatives of neuron outputs");
		for (SimpleMatrix matrix : neuronOutputs)
		{
			SimpleMatrix derivMatrix = new SimpleMatrix(matrix.getNumRows(), matrix.getNumCols()) ;
			derivMatrix = derivativeMatrix(matrix) ;
			derivMatrix.print() ;
		}
		System.out.println("=================\n") ;
	}

}
