package neural.network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class ANN2Test
{
    private static ANN2 ann ;
    private static int[] qtdNeuronsInLayer ;
    private static List<DataPoint> trainingData ;

    private static ANN2 ann2 = new ANN2(new int[] {2, 2, 2}, false, false) ;

    @BeforeAll
    static void beforeAll()
    {        
        qtdNeuronsInLayer = new int[] {2, 2, 2, 3} ;
        ann = new ANN2(qtdNeuronsInLayer, false, false) ;

        // trainingData is already normalized
        DataPoint DP1 = new DataPoint(List.of(1.0, 0.20833333333333), List.of(0.607759, 0.362069, 1.0)) ;
        trainingData = List.of(DP1) ;
    }

    @Test
    void testAnn2()
    {
        List<DataPoint> trainingData2 ;
        DataPoint DP1 = new DataPoint(List.of(0.05, 0.1), List.of(0.01, 0.99)) ;
        trainingData2 = List.of(DP1) ;

        List<SimpleMatrix> expectedWeights = new ArrayList<>() ;
        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.15, 0.2},
            {0.25, 0.3}
        })) ;
        expectedWeights.add(new SimpleMatrix(new double[][]
        {
            {0.4, 0.45},
            {0.5, 0.55}
        })) ;
        List<SimpleMatrix> expectedBiases = new ArrayList<>() ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.35},
            {0.35}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.6},
            {0.6}
        })) ;
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.05},
            {0.1}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.3775},
            {0.3925}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.10590596705977},
            {1.22492140409647}
        })) ;
        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.05},
            {0.1}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.593269992107187},
            {0.596884378259767}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.751365069552316},
            {0.772928465321463}
        })) ;
        double expectedAvrError = 0.298371108760003 ;
        List<SimpleMatrix> expectedDWeights = new ArrayList<>() ;
        expectedDWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0002307753, -0.000438569999999999},
            {-0.00024885999999999, -0.000497709999999985}
        })) ;
        expectedDWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0410835202, -0.041333814},
            {0.01130127, 0.011370121}
        })) ;

        ann2.forwardPropagation(trainingData2.get(0).getInputs());

        assertSimpleMatrixListEquals(expectedWeights, ann2.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, ann2.getBiases(), 1e-6);
        assertSimpleMatrixListEquals(expectedNeuronInputs, ann2.getNeuronInputs(), 1e-6);
        assertSimpleMatrixListEquals(expectedNeuronOutputs, ann2.getNeuronOutputs(), 1e-6);
        assertEquals(expectedAvrError, ann2.calcTotalError(trainingData2), 1e-6);

        ann2.backPropagationIteration(trainingData2.get(0)) ;
        assertSimpleMatrixListEquals(expectedDWeights, ann2.getdWeights(), 1e-6);

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
            {0.183485011316647},
            {0.440075880496294}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.637837761824192},
            {0.868641782097817}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.2933010708004},
            {1.56504657536641},
            {1.83679207993242}
        })) ;
        ann.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, ann.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.731058578630005},
            {0.55189576726823}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.545742989966838},
            {0.608277111401289}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.654264520296429},
            {0.704463002533615}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.784705405550395},
            {0.827076311464844},
            {0.862568870455212}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, ann.getNeuronOutputs(), 1e-6) ;
    }

    // @Test
    // void DO()
    // {
    //     DataPoint DP1 = trainingData.get(0) ;
    //     double expectedDO0 = -0.06489143 ;
    //     double expectedDO1 = -0.40245994 ;
    //     double expectedDO2 = 0.20951140 ;

    //     double DO0 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(0), ann.getOutputs().get(0), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(0)) ;
    //     double DO1 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(1), ann.getOutputs().get(1), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(1)) ;
    //     double DO2 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(2), ann.getOutputs().get(2), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(2)) ;

    //     assertEquals(expectedDO0, DO0, 1e-6) ;
    //     assertEquals(expectedDO1, DO1, 1e-6) ;
    //     assertEquals(expectedDO2, DO2, 1e-6) ;
    // }

    // @Test
    // void deltaMatrixToLastLayer()
    // {
    //     SimpleMatrix deltaMatrixToLastLayerOut0 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 0) ;
    //     SimpleMatrix deltaMatrixToLastLayerOut1 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 1) ;
    //     SimpleMatrix deltaMatrixToLastLayerOut2 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 2) ;

    //     double[][] expectedDeltaMatrixToLastLayerOut0 =
    //     {
    //         {1.0, 1.0},
    //         {0.0, 0.0},
    //         {0.0, 0.0}
    //     } ;

    //     double[][] expectedDeltaMatrixToLastLayerOut1 =
    //     {
    //         {0.0, 0.0},
    //         {1.0, 1.0},
    //         {0.0, 0.0}
    //     } ;

    //     double[][] expectedDeltaMatrixToLastLayerOut2 =
    //     {
    //         {0.0, 0.0},
    //         {0.0, 0.0},
    //         {1.0, 1.0}
    //     } ;

    //     for (int i = 0 ; i <= deltaMatrixToLastLayerOut0.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= deltaMatrixToLastLayerOut0.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedDeltaMatrixToLastLayerOut0[i][j], deltaMatrixToLastLayerOut0.get(i, j), 1e-6) ;
    //         }
    //     }

    //     for (int i = 0 ; i <= deltaMatrixToLastLayerOut1.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= deltaMatrixToLastLayerOut1.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedDeltaMatrixToLastLayerOut1[i][j], deltaMatrixToLastLayerOut1.get(i, j), 1e-6) ;
    //         }
    //     }

    //     for (int i = 0 ; i <= deltaMatrixToLastLayerOut2.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= deltaMatrixToLastLayerOut2.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedDeltaMatrixToLastLayerOut2[i][j], deltaMatrixToLastLayerOut2.get(i, j), 1e-6) ;
    //         }
    //     }
    // }

    // @Test
    // void deltaMatrixToBeforeLastLayer()
    // {        
    //     SimpleMatrix deltaMatrixToBeforeLastLayer = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 2, 0) ;

    //     double[][] expectedDeltaMatrixToBeforeLastLayer =
    //     {
    //         {-0.3321266, -0.3321266},
    //         {-0.3690296, -0.3690296}
    //     } ;

    //     for (int i = 0 ; i <= deltaMatrixToBeforeLastLayer.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= deltaMatrixToBeforeLastLayer.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedDeltaMatrixToBeforeLastLayer[i][j], deltaMatrixToBeforeLastLayer.get(i, j), 1e-6) ;
    //         }
    //     }
    // }

    // @Test
    // void deltaMatrixToDeepLayer()
    // {
    //     // ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 2, 0);
    //     // ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 3, 0);
    //     // ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 4, 0);
    //     List<SimpleMatrix> deltaMatrixToBeforeLastLayer = ann.getDeltaMatrices().get(qtdNeuronsInLayer.length - 4) ;

    //     double[][] expectedDeltaMatrixToBeforeLastLayer =
    //     {
    //         {0.7993272, 0.7993272},
    //         {0.9305707, 0.9305707}
    //     } ;

    //     // for (int i = 0 ; i <= deltaMatrixToBeforeLastLayer.getNumRows() - 1 ; i += 1)
    //     // {
    //     //     for (int j = 0 ; j <= deltaMatrixToBeforeLastLayer.getNumCols() - 1 ; j += 1)
    //     //     {
    //     //         assertEquals(expectedDeltaMatrixToBeforeLastLayer[i][j], deltaMatrixToBeforeLastLayer.get(i, j), 1e-6) ;
    //     //     }
    //     // }
    // }

    @Test
    void backPropagationIteration()
    {
        ann.forwardPropagation(trainingData.get(0).getInputs()) ;
        ann.backPropagationIteration(trainingData.get(0)) ;

        // add expected delta matrices for each layer
        List<List<SimpleMatrix>> expectedDeltaMatrices = new ArrayList<>() ;
        for (int layer = 0 ; layer <= qtdNeuronsInLayer.length - 1 ; layer += 1)
        {
            expectedDeltaMatrices.add(new ArrayList<>()) ;
        }
        expectedDeltaMatrices.get(0).add(new SimpleMatrix(new double[][]
        {
            {-0.0697288012231481, -0.0697288012231481},
            {-0.0819432658083923, -0.0819432658083923}
        })) ;
        expectedDeltaMatrices.get(0).add(new SimpleMatrix(new double[][]
        {
            {-0.197113329593845, -0.197113329593845},
            {-0.231692591310701, -0.231692591310701}
        })) ;
        expectedDeltaMatrices.get(0).add(new SimpleMatrix(new double[][]
        {
            {-0.402653864351401, -0.402653864351401},
            {-0.473363459968064, -0.473363459968064}
        })) ;

        expectedDeltaMatrices.get(1).add(new SimpleMatrix(new double[][]
        {
            {-0.341393930039757, -0.341393930039757},
            {-0.379326588933063, -0.379326588933063}
        })) ;
        expectedDeltaMatrices.get(1).add(new SimpleMatrix(new double[][]
        {
            {-0.97275662846969, -0.97275662846969},
            {-1.06118904923966, -1.06118904923966}
        })) ;
        expectedDeltaMatrices.get(1).add(new SimpleMatrix(new double[][]
        {
            {-1.99811698446105, -1.99811698446105},
            {-2.15181829095806, -2.15181829095806}
        })) ;
        expectedDeltaMatrices.get(2).add(new SimpleMatrix(new double[][]
        {
            {1.0, 1.0},
            {0.0, 0.0},
            {0.0, 0.0}
        })) ;
        expectedDeltaMatrices.get(2).add(new SimpleMatrix(new double[][]
        {
            {0.0, 0.0},
            {1.0, 1.0},
            {0.0, 0.0}
        })) ;
        expectedDeltaMatrices.get(2).add(new SimpleMatrix(new double[][]
        {
            {0.0, 0.0},
            {0.0, 0.0},
            {1.0, 1.0}
        })) ;

        for (int layer = 0 ; layer <= qtdNeuronsInLayer.length - 1 ; layer += 1)
        {
            assertSimpleMatrixListEquals(expectedDeltaMatrices.get(layer), ann.getDeltaMatrices().get(layer), 1e-6) ;
        }

        // add expected delta weights for each layer
        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00532955616179858, -0.00402342517151912},
            {-0.0103009955673974, -0.00777649837985515}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0300220966444396, -0.033462187441381},
            {-0.0164927546424299, -0.0183825817965285}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.043914546320999, -0.0472838923653939},
            {-0.2690448386764, -0.289687319104934},
            {0.138202546367729, 0.148806144535991}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaWeights, ann.getdWeights(), 1e-6) ;
    }

    // @Test
    // void backPropagation()
    // {
    //     ann.backPropagation(trainingData) ;
    //     List<SimpleMatrix> weights = ann.getWeights() ;
    //     double error = ann.calcOutputError(trainingData.get(0), 0) ;
    //     double expectedError = 0.000718980998 ;

    //     double[][] expectedWeightsLayer3 =
    //     {
    //         {0.8957975, 0.9954659},
    //         {1.1, 1.2},
    //         {1.3, 1.4}
    //     } ;
    //     double[][] expectedWeightsLayer2 =
    //     {
    //         {0.4978458, 0.5975236},
    //         {0.6974176, 0.7970313}
    //     } ;
    //     double[][] expectedWeightsLayer1 =
    //     {
    //         {0.0976922, 0.1971152},
    //         {0.2969113, 0.3961392}
    //     } ;

    //     SimpleMatrix weightsLayer3 = weights.get(2) ;
    //     for (int i = 0 ; i <= weightsLayer3.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= weightsLayer3.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedWeightsLayer3[i][j], weightsLayer3.get(i, j), 1e-6) ;
    //         }
    //     }

    //     SimpleMatrix weightsLayer2 = weights.get(1) ;
    //     for (int i = 0 ; i <= weightsLayer2.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= weightsLayer2.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedWeightsLayer2[i][j], weightsLayer2.get(i, j), 1e-6) ;
    //         }
    //     }

    //     SimpleMatrix weightsLayer1 = weights.get(0) ;
    //     for (int i = 0 ; i <= weightsLayer1.getNumRows() - 1 ; i += 1)
    //     {
    //         for (int j = 0 ; j <= weightsLayer1.getNumCols() - 1 ; j += 1)
    //         {
    //             assertEquals(expectedWeightsLayer1[i][j], weightsLayer1.get(i, j), 1e-6) ;
    //         }
    //     }

    //     assertEquals(expectedError, error, 1.e-6) ;
    // }

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
