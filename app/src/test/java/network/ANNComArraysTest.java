package network;

import org.junit.jupiter.api.Test;

class ANNComArraysTest
{
    @Test
    void matrix()
    {
    
        // SimpleMatrix mat = new SimpleMatrix(2, 2);
        // mat.set(0,0,1);
        // mat.set(0,1,2);
        // mat.set(1,0,3);
        // mat.set(1,1,4);
        
        // SimpleMatrix result = mat.mult(mat);
    }

    // @Test
    // void outputsAreZeroBeforeTraining()
    // {
    //     List<DataPoint> trainingDataPoints = new ArrayList<>() ;

    //     trainingDataPoints.add(new DataPoint(List.of(0.0), List.of(5.0))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.017453292519943295), List.of(4.999238475781956))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.03490658503988659), List.of(4.9969541350954785))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.05235987755982988), List.of(4.993147673772869))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.06981317007977318), List.of(4.987820251299121))) ;

    //     Data trainingData = new Data(trainingDataPoints) ;

    //     List<Double[]> expectedOutputs = new ArrayList<>() ;
    //     expectedOutputs.add(new Double[] {0.0}) ;
    //     expectedOutputs.add(new Double[] {0.0}) ;
    //     expectedOutputs.add(new Double[] {0.0}) ;
    //     expectedOutputs.add(new Double[] {0.0}) ;
    //     expectedOutputs.add(new Double[] {0.0}) ;

    //     ANN1 ann = new ANN1(trainingData.getNormalizedDataPoints(), false) ;
        
    //     assertTrue(areEqualWithTolerance(expectedOutputs, ann.getOutput()));

    // }

    // @Test
    // void outputsAreCorrectAfter1TrainingIteration()
    // {
    //     List<DataPoint> trainingDataPoints = new ArrayList<>() ;

    //     trainingDataPoints.add(new DataPoint(List.of(0.0), List.of(5.0))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.017453292519943295), List.of(4.999238475781956))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.03490658503988659), List.of(4.9969541350954785))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.05235987755982988), List.of(4.993147673772869))) ;
    //     trainingDataPoints.add(new DataPoint(List.of(0.06981317007977318), List.of(4.987820251299121))) ;

    //     Data trainingData = new Data(trainingDataPoints) ;

    //     List<Double[]> expectedOutputs = new ArrayList<>() ;
    //     expectedOutputs.add(new Double[] {0.650778}) ;
    //     expectedOutputs.add(new Double[] {0.666691}) ;
    //     expectedOutputs.add(new Double[] {0.679383}) ;
    //     expectedOutputs.add(new Double[] {0.684041}) ;
    //     expectedOutputs.add(new Double[] {0.675716}) ;
    //     double[][] expectedWeights = new double[][] {{0.494910}, {0.494910}} ;
    //     // double expectedError = 0.806445893385996 ;

    //     ANN1 ann = new ANN1(trainingData.getNormalizedDataPoints(), false) ;        
    //     ann.runTrainingOnce(trainingData.getNormalizedDataPoints()) ;

    //     areEqualWithTolerance(expectedOutputs, ann.getOutput()) ;
    //     assertTrue(areEqualWithTolerance(expectedWeights, ann.getWeights()[0])) ;
    //     // assertEquals(expectedError, ann.getError(), Math.pow(10, -6));
    // }

    // private boolean areEqualWithTolerance(List<Double[]> expected, List<Double[]> actual)
    // {
    //     if (expected.size() != actual.size()) { return false ;}

    //     return IntStream.range(0, expected.size()).allMatch(i -> areEqualWithTolerance(
    //                                                                             Arrays.stream(expected.get(i)).mapToDouble(Double::doubleValue).toArray(),
    //                                                                             Arrays.stream(actual.get(i)).mapToDouble(Double::doubleValue).toArray()
    //     )) ;
    // }

    // private boolean areEqualWithTolerance(double[] expected, double[] actual)
    // {
    //     double tolerance = Math.pow(10, -6) ;

    //     if (expected.length != actual.length) { return false ;}

    //     return IntStream.range(0, expected.length).allMatch(i -> Math.abs(expected[i] - actual[i]) <= tolerance) ;
    // }

    // private boolean areEqualWithTolerance(double[][] expected, double[][] actual)
    // {
    //     double tolerance = Math.pow(10, -6) ;

    //     if (expected.length != actual.length) {return false ;}

    //     for (int i = 0; i < expected.length; i++)
    //     {
    //         if (expected[i].length != actual[i].length) {return false ;}

    //         for (int j = 0; j < expected[i].length; j++)
    //         {
    //             if (tolerance < Math.abs(expected[i][j] - actual[i][j])) {return false ;}
    //         }
    //     }

    //     return true;
    // }
}
