package network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class ANNMatricialTest
{
    private static ANNMatricial ann ;
    private static ANNMatricial annComBias ;
    private static int[] qtdNeuronsInLayer ;
    private static List<DataPoint> trainingData ;

    @BeforeAll
    static void beforeAll()
    {        
        qtdNeuronsInLayer = new int[] {2, 2, 2, 3} ;
        ann = new ANNMatricial(qtdNeuronsInLayer, false, false, false) ;
        annComBias = new ANNMatricial(qtdNeuronsInLayer, false, false, false) ;
        annComBias.activateBiases() ;

        // trainingData is already normalized
        DataPoint DP1 = new DataPoint(List.of(1.0, 0.20833333333333), List.of(0.607759, 0.362069, 1.0)) ;
        trainingData = List.of(DP1) ;
    }

    @Test
    void initialParametersAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, ann.getQtdNeuronsInLayer());

        List<SimpleMatrix> expectedWeights = new ArrayList<>() ;

        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.1, 0.2},
            {0.3, 0.4}
        })) ;

        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.5, 0.6},
            {0.7, 0.8}
        })) ;

        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.9, 1.0},
            {1.1, 1.2},
            {1.3, 1.4}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, ann.getWeights(), 1e-6);
    }

    @Test
    void forwardPropagation()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.0},
            {0.20833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.141666666666666},
            {0.383333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.624484862385692},
            {0.850491734933162}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.28678436978467},
            {1.55716600118297},
            {1.82754763258126}
        })) ;
        ann.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, ann.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.0},
            {0.20833333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.535357552567207},
            {0.594676810170147}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.651237872067781},
            {0.700670284923671}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.78360241310259},
            {0.825946316413328},
            {0.861469320822647}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, ann.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIteration()
    {
        ann.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 0.132646909843869 ;

        assertEquals(expectedTotalError, ann.calcTotalError(trainingData), 1e-6) ;
        ann.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00269463712057808, -0.000561382733453759},
            {-0.0030456390989975, -0.00063450814562447}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0047846846339336, -0.00531484235560599},
            {-0.00486719362712479, -0.00540649359811859}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00970921092603497, -0.010446191595597},
            {-0.021714399740627, -0.0233626379941676},
            {0.00538321091037213, 0.00579182520574}
        })) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, ann.getdWeights(), 1e-6) ;
    }


    @Test
    void initialParametersComBiasAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, annComBias.getQtdNeuronsInLayer());

        List<SimpleMatrix> expectedWeights = new ArrayList<>() ;
        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.1, 0.2},
            {0.3, 0.4}
        })) ;

        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.5, 0.6},
            {0.7, 0.8}
        })) ;

        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.9, 1.0},
            {1.1, 1.2},
            {1.3, 1.4}
        })) ;
        
        List<SimpleMatrix> expectedBiases = new ArrayList<>() ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.1},
            {0.2}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.3},
            {0.4}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.5},
            {0.6},
            {0.7}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, annComBias.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, annComBias.getBiases(), 1e-6);
    }

    @Test
    void forwardPropagationComBias()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.0},
            {0.20833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.241666666666666},
            {0.583333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.965162593103236},
            {1.30555426854098}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.93850709889501},
            {2.34069160368839},
            {2.74287610848177}
        })) ;
        annComBias.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, annComBias.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.0},
            {0.20833333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.560124332099995},
            {0.641834045088731}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.724154250719015},
            {0.786768273247894}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.874188040982006},
            {0.912191497154539},
            {0.939509756072204}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annComBias.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationComBias()
    {
        annComBias.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 0.188639132682291 ;

        assertEquals(expectedTotalError, annComBias.calcTotalError(trainingData), 1e-6) ;
        annComBias.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00198515543984207, -0.000413574049967091},
            {-0.00216296200396094, -0.000450617084158523}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00393696576781461, -0.00451128172678849},
            {-0.00363501324907516, -0.00416528103476236}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0106098546898656, -0.0115272361454322},
            {-0.0159544958896333, -0.0173340019328258},
            {0.00124472397445714, 0.00135234907629354}
        })) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00198515543984207},
            {-0.00216296200396094}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00702873548280666},
            {-0.00648965424416919}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.0146513739018048},
            {-0.0220319025591468},
            {0.00171886579857986}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaWeights, annComBias.getdWeights(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaBiases, annComBias.getdBiases(), 1e-6) ;
    }

    private void assertSimpleMatrixListEquals(List<SimpleMatrix> expected, List<SimpleMatrix> actual, double tolerance)
    {
        assertEquals(expected.size(), actual.size(), "O número de matrizes não corresponde.");

        for (int i = 0; i <= expected.size() - 1; i++)
        {
            SimpleMatrix expectedMatrix = expected.get(i);
            SimpleMatrix actualMatrix = actual.get(i);

            assertEquals(expectedMatrix.getNumRows(), actualMatrix.getNumRows(), "Número de linhas não corresponde na matriz " + i);
            assertEquals(expectedMatrix.getNumCols(), actualMatrix.getNumCols(), "Número de colunas não corresponde na matriz " + i);

            for (int row = 0; row <= expectedMatrix.getNumRows() - 1; row++)
            {
                for (int col = 0; col <= expectedMatrix.getNumCols() - 1; col++)
                {
                    assertEquals(expectedMatrix.get(row, col), actualMatrix.get(row, col), tolerance, 
                        "Valores diferentes na matriz " + i + " na posição (" + row + ", " + col + ")");
                }
            }
        }
    }
}
