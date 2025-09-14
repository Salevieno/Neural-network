package neural.network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class DataTest
{
    
	protected static final Data trainingData = new Data("input.json") ;

    @Test
    void createData()
    {
        assertNotNull(trainingData);
        assertTrue(!trainingData.getListAllInputs().isEmpty());
        assertTrue(!trainingData.getListAllTargets().isEmpty());
        assertEquals(-10.0, trainingData.getMinInput());
        assertEquals(50.0, trainingData.getMaxInput());
        assertEquals(-6.3, trainingData.getMinTarget());
        assertEquals(5.3, trainingData.getMaxTarget());
    }

    @Test
    void normalizeInputsAndTargets()
    {
        assertEquals(Data.MIN_NORMALIZED_VALUE, trainingData.getNormalizedListAllInputs().stream().mapToDouble(Double::valueOf).min().getAsDouble()) ;
        assertEquals(Data.MAX_NORMALIZED_VALUE, trainingData.getNormalizedListAllInputs().stream().mapToDouble(Double::valueOf).max().getAsDouble()) ;
        assertEquals(Data.MIN_NORMALIZED_VALUE, trainingData.getNormalizedListAllTargets().stream().mapToDouble(Double::valueOf).min().getAsDouble()) ;
        assertEquals(Data.MAX_NORMALIZED_VALUE, trainingData.getNormalizedListAllTargets().stream().mapToDouble(Double::valueOf).max().getAsDouble()) ;
    }

    @Test
    void unormalizeOutputs()
    {
        double[] normalizedOutputs = new double[] {0.0, 0.15, 0.87, 1.0} ;
        double[] expectedResult = new double[] {-6.3, -4.56, 3.792, 5.3} ;
        double[] unormalizedOutputs = trainingData.unormalizeOutputs(normalizedOutputs) ;

        assertArrayEquals(expectedResult, unormalizedOutputs, 1e-6) ;
    }
}
