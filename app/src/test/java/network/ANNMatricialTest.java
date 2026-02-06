package network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import activationFunctions.None;
import activationFunctions.ReLu;

public class ANNMatricialTest
{
    private static ANNMatricial ann ;
    private static ANNMatricial annComBias ;
    private static ANNMatricial annComBiasProfundo ;
    private static ANNMatricial annComBiasProfundoActPrimeiraCamada ;
    private static ANNMatricial annReLu ;
    private static ANNMatricial annNoAct ;
    private static List<DataPoint> trainingData ;

    @BeforeAll
    static void beforeAll()
    {
        // trainingData is already normalized
        DataPoint DP1 = new DataPoint(List.of(1.0, 0.20833333333333), List.of(0.607759, 0.362069, 1.0)) ;
        trainingData = List.of(DP1) ;
    }

    @BeforeEach
    void beforeEach()
    {        
        int[] qtdNeuronsInLayer = new int[] {2, 2, 2, 3} ;
        ann = new ANNMatricial(qtdNeuronsInLayer, false, false, false) ;
        annComBias = new ANNMatricial(qtdNeuronsInLayer, false, false, false, false) ;
        annComBias.activateBiases() ;
        annComBiasProfundo = new ANNMatricial(qtdNeuronsInLayer, false, false, false) ;
        annComBiasProfundo.activateBiases() ;
        annComBiasProfundoActPrimeiraCamada = new ANNMatricial(qtdNeuronsInLayer, false, false, false) ;
        annComBiasProfundoActPrimeiraCamada.activateBiases() ;
        annComBiasProfundoActPrimeiraCamada.activateActFunctionOnFirstLayer() ;
        annReLu = new ANNMatricial(qtdNeuronsInLayer, new ReLu(), false, false, false, true) ;
        annReLu.activateBiases() ;
        annNoAct = new ANNMatricial(qtdNeuronsInLayer, new None(), false, false, false, true) ;
        annNoAct.activateBiases() ;
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


    // Testes com bias

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
            {0.0},
            {0.0}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.3},
            {0.4}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.5},
            {0.6}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.7},
            {0.8},
            {0.9}
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
            {0.441666666666666},
            {0.783333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.21616689328909},
            {1.57517772746265}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {2.22276994343384},
            {2.6427516979302},
            {3.06273345242655}
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
            {0.608656092315871},
            {0.686398078551927}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.771388290479262},
            {0.828520482002509}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.902275705702054},
            {0.93356283746702},
            {0.955329093465996}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annComBias.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationComBias()
    {
        annComBias.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 0.20767039304547 ;
        assertEquals(expectedTotalError, annComBias.calcTotalError(trainingData), 1e-6) ;

        annComBias.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {0.0},
            {0.0}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00140863805126431},
            {-0.00148731551482099}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00528023810258553},
            {-0.00467673049057136}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.0129843957910617},
            {-0.0177229571400153},
            {0.00095317477408925}
        })) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00140863805126431, -0.00029346626068006},
            {-0.00148731551482099, -0.000309857398921035}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00321384909001707, -0.00362434528791138},
            {-0.00284652050520565, -0.0032100988226334}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0100160108721732, -0.0107578378593218},
            {-0.0136712816104737, -0.0146838329921553},
            {0.000735267859512664, 0.000789724823261058}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaBiases, annComBias.getdBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, annComBias.getdWeights(), 1e-6) ;

        List<SimpleMatrix> expectedBiasesAfterUpdate = new ArrayList<>() ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.0},
            {0.0}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.298591361948736},
            {0.398512684485179}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.494719761897414},
            {0.595323269509429}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.687015604208938},
            {0.782277042859985},
            {0.900953174774089}
        })) ;

        List<SimpleMatrix> expectedWeightsAfterUpdate = new ArrayList<>() ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.0985913619487357, 0.19970653373932},
            {0.298512684485179, 0.399690142601079}
        })) ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.496786150909983, 0.596375654712089},
            {0.697153479494794, 0.796789901177367}
        })) ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.889983989127827, 0.989242162140678},
            {1.08632871838953, 1.18531616700784},
            {1.30073526785951, 1.40078972482326}
        })) ;

        assertSimpleMatrixListEquals(expectedBiasesAfterUpdate, annComBias.getBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedWeightsAfterUpdate, annComBias.getWeights(), 1e-6) ;
    }


    // Testes com bias profundo
    
    @Test
    void initialParametersComBiasProfundoAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, annComBiasProfundo.getQtdNeuronsInLayer());
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
            {0.6}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.7},
            {0.8},
            {0.9}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, annComBiasProfundo.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, annComBiasProfundo.getBiases(), 1e-6);
    }

    @Test
    void forwardPropagationComBiasProfundo()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.491666666666666},
            {0.893333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.23599603463981},
            {1.60201134544469}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {2.22967897822868},
            {2.65111209981945},
            {3.07254522141021}
        })) ;
        annComBiasProfundo.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, annComBiasProfundo.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.620498977748359},
            {0.709577576276054}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.774866297251501},
            {0.83229931070233}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.902883213822618},
            {0.934079501106121},
            {0.955745949113566}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annComBiasProfundo.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationComBiasProfundo()
    {
        annComBiasProfundo.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 0.208126367989977 ;

        assertEquals(expectedTotalError, annComBiasProfundo.calcTotalError(trainingData), 1e-6) ;
        annComBiasProfundo.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.000555584839490886},
            {-0.000831836119528092}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00136584500310366},
            {-0.0013966677972684}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00519861948978896},
            {-0.0045727848617994}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.0129390004646677},
            {-0.0176107695045008},
            {0.000935876478194591}
        })) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00150242950341402, -0.000557720042933989},
            {-0.00153633457699524, -0.000570306017217926}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00322573807911675, -0.00368882381754591},
            {-0.0028374083322097, -0.00324474559906745}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0100259953801925, -0.01076912116792},
            {-0.0136459917577022, -0.0146574313195337},
            {0.000725179141343418, 0.000778929347703882}
        })) ;
//         System.out.println(expectedDeltaBiases);
// System.out.println(annComBiasProfundo.getdBiases());
// System.out.println("Neuron inputs and outputs");
//         System.out.println(annComBiasProfundo.getNeuronInputs());
//         System.out.println("------");
//         System.out.println(annComBiasProfundo.getNeuronOutputs());
//         System.out.println("-----\n\n");
        assertSimpleMatrixListEquals(expectedDeltaBiases, annComBiasProfundo.getdBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, annComBiasProfundo.getdWeights(), 1e-6) ;

        List<SimpleMatrix> expectedBiasesAfterUpdate = new ArrayList<>() ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.0994444151605091},
            {0.199168163880472}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.298634154996896},
            {0.398603332202732}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.494801380510211},
            {0.595427215138201}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.687060999535332},
            {0.782389230495499},
            {0.900935876478195}
        })) ;

        List<SimpleMatrix> expectedWeightsAfterUpdate = new ArrayList<>() ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.098497570496586, 0.199442279957066},
            {0.298463665423005, 0.399429693982782}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.496774261920883, 0.596311176182454},
            {0.69716259166779, 0.796755254400933}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.889974004619808, 0.98923087883208},
            {1.0863540082423, 1.18534256868047},
            {1.30072517914134, 1.4007789293477}
        })) ;

        assertSimpleMatrixListEquals(expectedBiasesAfterUpdate, annComBiasProfundo.getBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedWeightsAfterUpdate, annComBiasProfundo.getWeights(), 1e-6) ;
    }
    

    // Testes com bias profundo e função de ativação na primeira camada
    
    @Test
    void initialParametersComBiasProfundoEAtivacaoNaPrimeiraCamadaAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, annComBiasProfundoActPrimeiraCamada.getQtdNeuronsInLayer());
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
            {0.6}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.7},
            {0.8},
            {0.9}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, annComBiasProfundoActPrimeiraCamada.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, annComBiasProfundoActPrimeiraCamada.getBiases(), 1e-6);
    }

    @Test
    void forwardPropagationComBiasProfundoEAtivacaoNaPrimeiraCamada()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.495163645799076},
            {0.865353302157664}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.2329277880373},
            {1.59794778733954}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {2.22862889997115},
            {2.64984129186155},
            {3.07105368375194}
        })) ;
        annComBiasProfundoActPrimeiraCamada.forwardPropagation(trainingData.get(0).getInputs()) ;
        System.out.println(annComBiasProfundoActPrimeiraCamada.getNeuronOutputs());
        assertSimpleMatrixListEquals(expectedNeuronInputs, annComBiasProfundoActPrimeiraCamada.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.750260105595118},
            {0.600688176197821}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.621322098694228},
            {0.703777897816976}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.774330594808433},
            {0.831731364643558}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.902791098627087},
            {0.934001207944538},
            {0.955682820688223}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annComBiasProfundoActPrimeiraCamada.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationComBiasProfundoEAtivacaoNaPrimeiraCamada()
    {
        annComBiasProfundoActPrimeiraCamada.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 0.208057201043385 ;

        assertEquals(expectedTotalError, annComBiasProfundoActPrimeiraCamada.calcTotalError(trainingData), 1e-6) ;
        annComBiasProfundoActPrimeiraCamada.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.000105302547102956},
            {-0.000201624992808101}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00136874580976963},
            {-0.00141709673480333}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.00521115317277652},
            {-0.00458845529578939}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-0.0129459097798681},
            {-0.0176277946785734},
            {0.000938486446620895}
        })) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00102691537577063, -0.000822189424148926},
            {-0.00106319114589205, -0.000851233253124902}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.00323780462592659, -0.00366749442513892},
            {-0.00285090867414451, -0.00322925342229783}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-0.0100244140201816, -0.0107675192077621},
            {-0.0136497407386207, -0.0146615897236663},
            {0.00072669876843161, 0.000780568612947481}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaBiases, annComBiasProfundoActPrimeiraCamada.getdBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, annComBiasProfundoActPrimeiraCamada.getdWeights(), 1e-6) ;

        List<SimpleMatrix> expectedBiasesAfterUpdate = new ArrayList<>() ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.0998946974528971},
            {0.199798375007192}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.29863125419023},
            {0.398582903265197}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.494788846827223},
            {0.595411544704211}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.687054090220132},
            {0.782372205321427},
            {0.900938486446621}
        })) ;

        List<SimpleMatrix> expectedWeightsAfterUpdate = new ArrayList<>() ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.0989730846242294, 0.199177810575851},
            {0.298936808854108, 0.399148766746875}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.496762195374073, 0.596332505574861},
            {0.697149091325856, 0.796770746577702}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {0.889975585979818, 0.989232480792238},
            {1.08635025926138, 1.18533841027633},
            {1.30072669876843, 1.40078056861295}
        })) ;

        assertSimpleMatrixListEquals(expectedBiasesAfterUpdate, annComBiasProfundoActPrimeiraCamada.getBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedWeightsAfterUpdate, annComBiasProfundoActPrimeiraCamada.getWeights(), 1e-6) ;
    }
    

    // Testes com ReLu e bias profundo

    @Test
    void initialParametersReLuComBiasProfundoAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, annReLu.getQtdNeuronsInLayer());
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
            {0.6}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.7},
            {0.8},
            {0.9}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, annReLu.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, annReLu.getBiases(), 1e-6);
    }

    @Test
    void forwardPropagationReLuComBiasProfundo()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.491666666666666},
            {0.893333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.28183333333333},
            {1.65883333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {3.51248333333333},
            {4.20061666666666},
            {4.88875}
        })) ;
        annReLu.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, annReLu.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.491666666666666},
            {0.893333333333332}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.28183333333333},
            {1.65883333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {3.51248333333333},
            {4.20061666666666},
            {4.88875}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annReLu.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationReLuComBiasProfundo()
    {
        annReLu.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 19.1471241022154 ;

        assertEquals(expectedTotalError, annReLu.calcTotalError(trainingData), 1e-6) ;
        annReLu.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-3.375644261},
            {-5.00134823933333}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-7.50733836999999},
            {-8.74970141333333}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-5.94601466666666},
            {-6.47761576666666}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-1.45236216666667},
            {-1.91927383333333},
            {-1.944375}
        })) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-8.25807220699999, -3.06549650108331},
            {-9.62467155466666, -3.57279474377775}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-2.92345721111111, -5.31177310222221},
            {-3.18482775194444, -5.78667008488888}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-1.86168623730555, -2.40922677413889},
            {-2.46018917536111, -3.18375541052777},
            {-2.4923646875, -3.2253940625}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaBiases, annReLu.getdBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, annReLu.getdWeights(), 1e-6) ;

        List<SimpleMatrix> expectedBiasesAfterUpdate = new ArrayList<>() ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-3.275644261},
            {-4.80134823933333}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-7.20733836999999},
            {-8.34970141333333}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-5.44601466666666},
            {-5.87761576666666}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-0.752362166666666},
            {-1.11927383333333},
            {-1.044375}
        })) ;

        List<SimpleMatrix> expectedWeightsAfterUpdate = new ArrayList<>() ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-8.158072207, -2.86549650108331},
            {-9.32467155466666, -3.17279474377775}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-2.42345721111111, -4.71177310222221},
            {-2.48482775194444, -4.98667008488888}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-0.961686237305553, -1.40922677413889},
            {-1.36018917536111, -1.98375541052777},
            {-1.1923646875, -1.8253940625}
        })) ;

        assertSimpleMatrixListEquals(expectedBiasesAfterUpdate, annReLu.getBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedWeightsAfterUpdate, annReLu.getWeights(), 1e-6) ;
    }


    // Testes sem função de ativação e com bias profundo

    @Test
    void initialParametersNoActComBiasProfundoAreCorrect()
    {
        int[] qtdNeuronsPerLayerExpected = new int[] {trainingData.get(0).getInputs().size(), 2, 2, trainingData.get(0).getTargets().size()} ;

        assertArrayEquals(qtdNeuronsPerLayerExpected, annNoAct.getQtdNeuronsInLayer());
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
            {0.6}
        })) ;
        expectedBiases.add(new SimpleMatrix(new double[][]
        {
            {0.7},
            {0.8},
            {0.9}
        })) ;

        assertSimpleMatrixListEquals(expectedWeights, annNoAct.getWeights(), 1e-6);
        assertSimpleMatrixListEquals(expectedBiases, annNoAct.getBiases(), 1e-6);
    }

    @Test
    void forwardPropagationNoActComBiasProfundo()
    {
        List<SimpleMatrix> expectedNeuronInputs = new ArrayList<>() ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {0.491666666666666},
            {0.893333333333332}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {1.28183333333333},
            {1.65883333333333}
        })) ;
        expectedNeuronInputs.add(new SimpleMatrix(new double[][]
        {
            {3.51248333333333},
            {4.20061666666666},
            {4.88875}
        })) ;
        annNoAct.forwardPropagation(trainingData.get(0).getInputs()) ;
        assertSimpleMatrixListEquals(expectedNeuronInputs, annNoAct.getNeuronInputs(), 1e-6) ;

        List<SimpleMatrix> expectedNeuronOutputs = new ArrayList<>() ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.1},
            {0.40833333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {0.491666666666666},
            {0.893333333333332}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {1.28183333333333},
            {1.65883333333333}
        })) ;
        expectedNeuronOutputs.add(new SimpleMatrix(new double[][]
        {
            {3.51248333333333},
            {4.20061666666666},
            {4.88875}
        })) ;
        assertSimpleMatrixListEquals(expectedNeuronOutputs, annNoAct.getNeuronOutputs(), 1e-6) ;
    }

    @Test
    void backPropagationIterationNoActComBiasProfundo()
    {
        annNoAct.forwardPropagation(trainingData.get(0).getInputs()) ;
        double expectedTotalError = 19.1471241022154 ;

        assertEquals(expectedTotalError, annNoAct.calcTotalError(trainingData), 1e-6) ;
        annNoAct.backPropagationIteration(trainingData.get(0)) ;

        List<SimpleMatrix> expectedDeltaBiases = new ArrayList<>() ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-3.375644261},
            {-5.00134823933333}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-7.50733836999999},
            {-8.74970141333333}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-5.94601466666666},
            {-6.47761576666666}
        })) ;
        expectedDeltaBiases.add(new SimpleMatrix(new double[][]
        {
            {-1.45236216666667},
            {-1.91927383333333},
            {-1.944375}
        })) ;

        List<SimpleMatrix> expectedDeltaWeights = new ArrayList<>() ;
        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-8.25807220699999, -3.06549650108331},
            {-9.62467155466666, -3.57279474377775}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-2.92345721111111, -5.31177310222221},
            {-3.18482775194444, -5.78667008488888}
        })) ;

        expectedDeltaWeights.add(new SimpleMatrix(new double[][]
        {
            {-1.86168623730555, -2.40922677413889},
            {-2.46018917536111, -3.18375541052777},
            {-2.4923646875, -3.2253940625}
        })) ;

        assertSimpleMatrixListEquals(expectedDeltaBiases, annNoAct.getdBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedDeltaWeights, annNoAct.getdWeights(), 1e-6) ;

        List<SimpleMatrix> expectedBiasesAfterUpdate = new ArrayList<>() ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-3.275644261},
            {-4.80134823933333}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-7.20733836999999},
            {-8.34970141333333}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-5.44601466666666},
            {-5.87761576666666}
        })) ;
        expectedBiasesAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-0.752362166666666},
            {-1.11927383333333},
            {-1.044375}
        })) ;

        List<SimpleMatrix> expectedWeightsAfterUpdate = new ArrayList<>() ;
        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-8.158072207, -2.86549650108331},
            {-9.32467155466666, -3.17279474377775}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-2.42345721111111, -4.71177310222221},
            {-2.48482775194444, -4.98667008488888}
        })) ;

        expectedWeightsAfterUpdate.add(new SimpleMatrix(new double[][]
        {
            {-0.961686237305553, -1.40922677413889},
            {-1.36018917536111, -1.98375541052777},
            {-1.1923646875, -1.8253940625}
        })) ;

        assertSimpleMatrixListEquals(expectedBiasesAfterUpdate, annNoAct.getBiases(), 1e-6) ;
        assertSimpleMatrixListEquals(expectedWeightsAfterUpdate, annNoAct.getWeights(), 1e-6) ;
    }


    // Funções auxiliares

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
