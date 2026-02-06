package network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;

import activationFunctions.ActivationFunction;
import activationFunctions.Sigmoid;

public class ANNMatricial extends ANN
{
	private List<SimpleMatrix> neuronInputs ;
	protected List<SimpleMatrix> neuronOutputs ;
	protected List<SimpleMatrix> weights ;
	private List<SimpleMatrix> dWeights ;
	protected List<SimpleMatrix> biases ;
	protected List<SimpleMatrix> dBiases ;
	private List<List<SimpleMatrix>> deltaMatrices ;

    public ANNMatricial(int[] qtdNeuronsInLayer, ActivationFunction activationFunction, boolean randomizeInitialWeights, boolean randomizeInitialBiases, boolean adaptativeLearningRate, boolean biasOnFirstLayer)
    {
		super(qtdNeuronsInLayer, activationFunction, adaptativeLearningRate) ;
		this.biasOnFirstLayer = biasOnFirstLayer ;
		this.neuronInputs = new ArrayList<>() ;
		this.neuronOutputs = new ArrayList<>() ;
		this.qtdNeuronsInLayer = qtdNeuronsInLayer ;
		for (int neuron : qtdNeuronsInLayer)
		{
			this.neuronInputs.add(new SimpleMatrix(neuron, 1)) ;
			this.neuronOutputs.add(new SimpleMatrix(neuron, 1)) ;
		}
		this.weights = initWeights(qtdLayers, randomizeInitialWeights) ;
		this.dWeights = initDWeights(qtdLayers) ;
		this.biases = initBiases(qtdLayers, randomizeInitialBiases) ;
		this.dBiases = initDBiases(qtdLayers) ;
		this.deltaMatrices = new ArrayList<>() ;
    }

    public ANNMatricial(int[] qtdNeuronsInLayer, boolean randomizeInitialWeights, boolean randomizeInitialBiases, boolean adaptativeLearningRate, boolean biasOnFirstLayer)
    {
		this(qtdNeuronsInLayer, new Sigmoid(), randomizeInitialWeights, randomizeInitialBiases, adaptativeLearningRate, biasOnFirstLayer) ;
    }

    public ANNMatricial(int[] qtdNeuronsInLayer, boolean randomizeInitialWeights, boolean randomizeInitialBiases, boolean adaptativeLearningRate)
    {
		this(qtdNeuronsInLayer, randomizeInitialWeights, randomizeInitialBiases, adaptativeLearningRate, true) ;
    }

    public ANNMatricial(boolean randomizeInitialWeights, boolean randomizeInitialBiases, boolean adaptativeLearningRate)
    {
        this(STD_QTD_NEURONS, randomizeInitialWeights, randomizeInitialBiases, adaptativeLearningRate) ;
    }

    public ANNMatricial()
    {
        this(true, true, false) ;
    }

	private List<SimpleMatrix> initWeights(int Nlayers, boolean randomInitialWeights)
	{
		List<SimpleMatrix> weights = new ArrayList<>() ;

		double startValue = 0.1 ;
		final double inc = 0.1 ;
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

	private List<SimpleMatrix> initDWeights(int Nlayers)
	{
		List<SimpleMatrix> deltaWeights = new ArrayList<>() ;

		for (int i = 0; i <= Nlayers - 2; i += 1)
		{
			deltaWeights.add(new SimpleMatrix(qtdNeuronsInLayer[i + 1], qtdNeuronsInLayer[i])) ;
		}
		return deltaWeights ;
	}

	private List<SimpleMatrix> initDBiases(int Nlayers)
	{
		List<SimpleMatrix> deltaBiases = new ArrayList<>() ;

		for (int i = 0; i <= Nlayers - 1; i += 1)
		{
			deltaBiases.add(new SimpleMatrix(qtdNeuronsInLayer[i], 1)) ;
		}
		return deltaBiases ;
	}

	private List<SimpleMatrix> initBiases(int Nlayers, boolean randomInitialBiases)
	{
		List<SimpleMatrix> biases = new ArrayList<>() ;

		double startValue = 0.1 ;
		final double inc = 0.1 ;
		for (int layer = 0; layer <= Nlayers - 1; layer += 1)
		{
			int qtdNeuronsCurrentLayer = qtdNeuronsInLayer[layer] ;

			biases.add(SimpleMatrix.random_DDRM(qtdNeuronsCurrentLayer, 1, -0.5, 0.5, new java.util.Random())) ;

			if (!randomInitialBiases)
			{
				for (int row = 0 ; row <= biases.get(layer).getNumRows() - 1 ; row += 1)
				{
					biases.get(layer).set(row, 0, startValue) ;
					startValue += inc ;
				}
			}
		}

		if (!biasOnFirstLayer)
		{
			biases.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1)) ;
		}

		return biases ;
	}

	public void forwardPropagation(List<Double> input)
	{
		neuronInputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().mapToDouble(Double::doubleValue).toArray())) ;
		neuronOutputs.set(0, new SimpleMatrix(qtdNeuronsInLayer[0], 1, true, input.stream().map(d -> d).mapToDouble(Double::valueOf).toArray())) ;
		// TODO tem que aplicar activation function na camada de entrada e SIM tem que ajustar todos os testes OU colocar uma condição na derivativeMatrix para não aplicar na camada de entrada

		if (biasIsActive && biasOnFirstLayer)
		{
			neuronInputs.set(0, neuronInputs.get(0).plus(biases.get(0))) ;
			neuronOutputs.set(0, neuronOutputs.get(0).plus(biases.get(0))) ;
		}
				
		if (actFunctionOnFirstLayer)
		{
			 neuronOutputs.set(0, neuronInputs.get(0).elementOp((SimpleOperations.ElementOpReal) (row, col, value) -> act.f(value)));
		}

		for (int layer = 1; layer <= qtdLayers - 1; layer += 1)
		{
			SimpleMatrix neuronsPrevLayer = neuronOutputs.get(layer - 1) ;
			SimpleMatrix newNeuronInputs = weights.get(layer - 1).mult(neuronsPrevLayer) ;

			if (biasIsActive)
			{
				newNeuronInputs = newNeuronInputs.plus(biases.get(layer)) ;
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
		SimpleMatrix DOMatrix = new SimpleMatrix(qtdNeuronsInLayer[qtdLayers - 1], 1) ;
		for (int outputID = 0 ; outputID <= qtdNeuronsInLayer[qtdLayers - 1] - 1 ; outputID += 1)
		{
			double DO = calcPointDError(dataPoint.getTargets().get(outputID), neuronOutputs.get(qtdLayers - 1).get(outputID)) ;
			DOMatrix.set(outputID, 0, DO) ;
		}
		
		deltaMatrices = new ArrayList<>(Collections.nCopies(qtdLayers, new ArrayList<>())) ;
		for (int layer = qtdLayers - 1 ; 0 <= layer ; layer += -1)
		{
			deltaMatrices.set(layer, new ArrayList<>()) ;
			SimpleMatrix sumDeltaMatrices = new SimpleMatrix(qtdNeuronsInLayer[layer], 1) ;
			SimpleMatrix deltaMatrix = calcDeltasToLayer(layer, DOMatrix) ;
			sumDeltaMatrices = sumDeltaMatrices.plus(deltaMatrix) ;

			deltaMatrices.get(layer).add(sumDeltaMatrices);
		}

		for (int layer = qtdLayers - 1 ; 1 <= layer ; layer += -1)
		{
			SimpleMatrix dBiasMatrix = calcDBias(layer, deltaMatrices.get(layer).get(0), dataPoint.getTargets()) ;
			dBiases.set(layer, dBiasMatrix) ;
			SimpleMatrix dWeightsMatrix = calcDWeights(layer, dBiasMatrix) ;
			dWeights.set(layer - 1, dWeightsMatrix) ;
		}

		if (biasIsActive && biasOnFirstLayer)
		{
			SimpleMatrix dBiasMatrix = calcDBias(0, deltaMatrices.get(0).get(0), dataPoint.getTargets()) ;
			dBiases.set(0, dBiasMatrix) ;
		}

		updateWeights(dWeights) ;
		updateBiases(dBiases) ;
	}

	private SimpleMatrix calcDWeights(int layer, SimpleMatrix dBiasMatrix)
	{
		SimpleMatrix neuronOutputsPrevLayer = neuronOutputs.get(layer - 1) ;
		SimpleMatrix dWeights = neuronOutputsPrevLayer.mult(dBiasMatrix.transpose()).transpose() ;

		return dWeights ;
	}

	private void updateWeights(List<SimpleMatrix> dWeights)
	{
		for (int i = 0 ; i <= weights.size() - 1 ; i+= 1)
		{
			weights.set(i, weights.get(i).plus(dWeights.get(i)));
		}
	}

	private SimpleMatrix calcDBias(int layer, SimpleMatrix deltaMatrix, List<Double> targets)
	{
		SimpleMatrix derivativeMatrix = derivativeMatrix(neuronInputs.get(layer)) ;

		if (!actFunctionOnFirstLayer && layer == 0)
		{
			derivativeMatrix = SimpleMatrix.ones(qtdNeuronsInLayer[layer], 1) ;
		}

		SimpleMatrix dBias = deltaMatrix.elementMult(derivativeMatrix).scale(learningRate) ;
		return dBias ;
	}

	private void updateBiases(List<SimpleMatrix> dBiases)
	{
		for (int i = 0 ; i <= biases.size() - 1 ; i+= 1)
		{
			biases.set(i, biases.get(i).plus(dBiases.get(i)));
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
		if (adaptativeLearningRate)
		{
			learningRate = updateLRate(learningRate, trainIterationError / trainingData.size(), LEARNING_RATE_MIN, LEARNING_RATE_MAX) ;
		}
	}

	public void test(List<DataPoint> testingDataPoints)
	{
		for (DataPoint dataPoint : testingDataPoints)
		{
			forwardPropagation(dataPoint.getInputs()) ;
			lastOutputsPerDataPoint.put(dataPoint, getOutputsAsList()) ;
		}
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

	private SimpleMatrix calcDeltasToLayer(int layer, SimpleMatrix DOMatrix)
	{
		SimpleMatrix deltaMatrix = new SimpleMatrix(qtdNeuronsInLayer[layer], 1) ;

		// last layer
		if (layer == qtdLayers - 1)
		{
			deltaMatrix = DOMatrix.copy() ;
			return deltaMatrix ;
		}

		// layers before that
		SimpleMatrix dInputsNextLayer = derivativeMatrix(neuronInputs.get(layer + 1)) ;
		SimpleMatrix weightMatrixNextLayer = weights.get(layer) ;
		SimpleMatrix deltaMatrixNextLayer = deltaMatrices.get(layer + 1).get(0) ;
		deltaMatrix = weightMatrixNextLayer.transpose().mult(dInputsNextLayer.elementMult(deltaMatrixNextLayer)) ;
		return deltaMatrix ;
	}

	public double calcOutputError(DataPoint dataPoint, int outputID)
	{
		return calcPointError(dataPoint.getTargets().get(outputID), getOutputs().get(outputID)) ;
	}

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
		return matrix.elementOp((SimpleOperations.ElementOpReal) (row, col, x) -> act.df(x)) ;
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
