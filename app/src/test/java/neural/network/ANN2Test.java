package neural.network;

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

    @BeforeAll
    static void beforeAll()
    {        
        qtdNeuronsInLayer = new int[] {0, 2, 2, 0} ;
        ann = new ANN2(qtdNeuronsInLayer, false, false) ;
    
        // trainingData is already normalized
        DataPoint DP1 = new DataPoint(List.of(1.0, 0.20833333333333), List.of(0.607759, 0.362069, 1.0)) ;
        DataPoint DP2 = new DataPoint(List.of(1.0, 0.20833333333333), List.of(0.607759, 0.362069, 1.0)) ;
        trainingData = List.of(DP1, DP2) ;
    
        ann.forwardPropagation(DP1.getInputs()) ;
    }

    @Test
    void DO()
    {
        DataPoint DP1 = trainingData.get(0) ;
        double expectedDO0 = -0.06489143 ;
        double expectedDO1 = -0.40245994 ;
        double expectedDO2 = 0.20951140 ;

        double DO0 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(0), ann.getOutputs().get(0), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(0)) ;
        double DO1 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(1), ann.getOutputs().get(1), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(1)) ;
        double DO2 = ann.calcOutputErrorPropagatedToLastLayer(DP1.getTargets().get(2), ann.getOutputs().get(2), ann.getNeuronInputs().get(ann.qtdLayers - 1).get(2)) ;

        assertEquals(expectedDO0, DO0, 1e-6) ;
        assertEquals(expectedDO1, DO1, 1e-6) ;
        assertEquals(expectedDO2, DO2, 1e-6) ;
    }

    @Test
    void forwardPropagation()
    {

    }

    @Test
    void deltaMatrixToLastLayer()
    {
        SimpleMatrix deltaMatrixToLastLayerOut0 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 0) ;
        SimpleMatrix deltaMatrixToLastLayerOut1 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 1) ;
        SimpleMatrix deltaMatrixToLastLayerOut2 = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 1, 2) ;

        double[][] expectedDeltaMatrixToLastLayerOut0 =
        {
            {1.0, 1.0},
            {0.0, 0.0},
            {0.0, 0.0}
        } ;

        double[][] expectedDeltaMatrixToLastLayerOut1 =
        {
            {0.0, 0.0},
            {1.0, 1.0},
            {0.0, 0.0}
        } ;

        double[][] expectedDeltaMatrixToLastLayerOut2 =
        {
            {0.0, 0.0},
            {0.0, 0.0},
            {1.0, 1.0}
        } ;

        for (int i = 0 ; i <= deltaMatrixToLastLayerOut0.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= deltaMatrixToLastLayerOut0.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedDeltaMatrixToLastLayerOut0[i][j], deltaMatrixToLastLayerOut0.get(i, j), 1e-6) ;
            }
        }

        for (int i = 0 ; i <= deltaMatrixToLastLayerOut1.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= deltaMatrixToLastLayerOut1.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedDeltaMatrixToLastLayerOut1[i][j], deltaMatrixToLastLayerOut1.get(i, j), 1e-6) ;
            }
        }

        for (int i = 0 ; i <= deltaMatrixToLastLayerOut2.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= deltaMatrixToLastLayerOut2.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedDeltaMatrixToLastLayerOut2[i][j], deltaMatrixToLastLayerOut2.get(i, j), 1e-6) ;
            }
        }
    }

    @Test
    void deltaMatrixToBeforeLastLayer()
    {        
        SimpleMatrix deltaMatrixToBeforeLastLayer = ann.calcDeltasToLayer(qtdNeuronsInLayer.length - 2, 0) ;

        double[][] expectedDeltaMatrixToBeforeLastLayer =
        {
            {-0.3321266, -0.3321266},
            {-0.3690296, -0.3690296}
        } ;

        for (int i = 0 ; i <= deltaMatrixToBeforeLastLayer.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= deltaMatrixToBeforeLastLayer.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedDeltaMatrixToBeforeLastLayer[i][j], deltaMatrixToBeforeLastLayer.get(i, j), 1e-6) ;
            }
        }
    }

    @Test
    void deltaMatrixToDeepLayer()
    {
        ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 2, 0);
        ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 3, 0);
        ann.updateDeltaMatrix(qtdNeuronsInLayer.length - 4, 0);
        SimpleMatrix deltaMatrixToBeforeLastLayer = ann.getDeltaMatrices().get(qtdNeuronsInLayer.length - 4) ;

        double[][] expectedDeltaMatrixToBeforeLastLayer =
        {
            {0.7993272, 0.7993272},
            {0.9305707, 0.9305707}
        } ;

        for (int i = 0 ; i <= deltaMatrixToBeforeLastLayer.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= deltaMatrixToBeforeLastLayer.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedDeltaMatrixToBeforeLastLayer[i][j], deltaMatrixToBeforeLastLayer.get(i, j), 1e-6) ;
            }
        }
    }

    @Test
    void backPropagationIteration()
    {
        ann.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> deltaMatrices = ann.getDeltaMatrices() ;
        List<SimpleMatrix> expectedDeltaMatrices = new ArrayList<>() ;

        // expectedDeltaMatrices.add(new SimpleMatrix(null)) ;
    }

    @Test
    void backPropagation()
    {
        ann.backPropagation(trainingData) ;
        List<SimpleMatrix> weights = ann.getWeights() ;
        double error = ann.calcOutputError(trainingData.get(0), 0) ;
        double expectedError = 0.000718980998 ;

        double[][] expectedWeightsLayer3 =
        {
            {0.8957975, 0.9954659},
            {1.1, 1.2},
            {1.3, 1.4}
        } ;
        double[][] expectedWeightsLayer2 =
        {
            {0.4978458, 0.5975236},
            {0.6974176, 0.7970313}
        } ;
        double[][] expectedWeightsLayer1 =
        {
            {0.0976922, 0.1971152},
            {0.2969113, 0.3961392}
        } ;

        SimpleMatrix weightsLayer3 = weights.get(2) ;
        for (int i = 0 ; i <= weightsLayer3.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= weightsLayer3.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedWeightsLayer3[i][j], weightsLayer3.get(i, j), 1e-6) ;
            }
        }

        SimpleMatrix weightsLayer2 = weights.get(1) ;
        for (int i = 0 ; i <= weightsLayer2.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= weightsLayer2.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedWeightsLayer2[i][j], weightsLayer2.get(i, j), 1e-6) ;
            }
        }

        SimpleMatrix weightsLayer1 = weights.get(0) ;
        for (int i = 0 ; i <= weightsLayer1.getNumRows() - 1 ; i += 1)
        {
            for (int j = 0 ; j <= weightsLayer1.getNumCols() - 1 ; j += 1)
            {
                assertEquals(expectedWeightsLayer1[i][j], weightsLayer1.get(i, j), 1e-6) ;
            }
        }

        assertEquals(expectedError, error, 1.e-6) ;
    }
}
