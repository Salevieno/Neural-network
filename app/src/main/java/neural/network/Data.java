package neural.network;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class Data
{

	private List<DataPoint> dataPoints ;
    private List<Double> listAllInputs ;
    private List<Double> listAllTargets ;
	private List<DataPoint> normalizedDataPoints ;
    private List<Double> normalizedListAllInputs ;
    private List<Double> normalizedListAllTargets ;
    private double minInput ;
    private double maxInput ;
    private double minTarget ;
    private double maxTarget ;

    public static final double MIN_NORMALIZED_VALUE = 0.0 ;
    public static final double MAX_NORMALIZED_VALUE = 1.0 ;

    public Data(List<DataPoint> dataPoints)
    {
		String validateMsg = validate(dataPoints) ;

		if (!"ok".equals(validateMsg)) { System.out.println(validateMsg) ; return ;}

        this.dataPoints = dataPoints ;
        this.listAllInputs = allInputs(dataPoints) ;
        this.listAllTargets = allTargets(dataPoints) ;
        this.minInput = listAllInputs.stream().mapToDouble(i -> i).min().getAsDouble() ;
        this.maxInput = listAllInputs.stream().mapToDouble(i -> i).max().getAsDouble() ;
        this.minTarget = listAllTargets.stream().mapToDouble(i -> i).min().getAsDouble() ;
        this.maxTarget = listAllTargets.stream().mapToDouble(i -> i).max().getAsDouble() ;
        this.normalizedListAllInputs = listAllInputs.stream().map(i -> normalize(minInput, i, maxInput)).toList() ;
        this.normalizedListAllTargets = listAllTargets.stream().map(i -> normalize(minTarget, i, maxTarget)).toList() ;
		this.normalizedDataPoints = new ArrayList<>() ;
		for (DataPoint dataPoint : dataPoints)
		{
			List<Double> normalizedInputs = dataPoint.getInputs().stream().map(i -> normalize(minInput, i, maxInput)).toList() ;
			List<Double> normalizedTargets = dataPoint.getTargets().stream().map(i -> normalize(minTarget, i, maxTarget)).toList() ;
			this.normalizedDataPoints.add(new DataPoint(normalizedInputs, normalizedTargets)) ;
		}

    }

    public Data(String filePath)
    {
        this(loadInput(filePath)) ;
    }

	private String validate(List<DataPoint> dataPoints)
	{
		if (dataPoints == null || dataPoints.isEmpty())
		{
			return "Null or empty dataPoints" ;
		}

		boolean hasInvalidDP = dataPoints.stream().anyMatch(dp -> dp.getInputs() == null || dp.getInputs().isEmpty() || dp.getTargets() == null || dp.getTargets().isEmpty()) ;
		
		return hasInvalidDP ? "Null or empty datapoint when trying to create Data" : "ok" ;
	}

	public static List<DataPoint> loadInput(String fileName)
	{
		List<DataPoint> trainingData = new ArrayList<>() ;
		// Load the JSON file from the resources folder
        InputStream inputStream = ANN2.class.getResourceAsStream("/data/" + fileName);
        if (inputStream == null)
		{
            return null ;
        }

		try
		{
			// Load the JSON file
			Reader reader = new InputStreamReader(inputStream);
	
			// Define the structure of the JSON
			Type dataType = new TypeToken<List<DataPoint>>() {}.getType();
	
			// Parse the JSON using Gson
			Gson gson = new Gson();
			trainingData = gson.fromJson(reader, dataType);
	
			reader.close();
	
			return trainingData ;
		}
		catch (Exception e)
		{
			e.printStackTrace();
			return null ;
		}
	}

	private static List<Double> allInputs(List<DataPoint> trainingData)
	{
		List<Double> allInputData = new ArrayList<>();
		for (DataPoint dataPoint : trainingData)
		{
			allInputData.addAll(dataPoint.getInputs());
		}
		return allInputData;
	}

	private static List<Double> allTargets(List<DataPoint> trainingData)
	{
		List<Double> allInputData = new ArrayList<>();
		for (DataPoint dataPoint : trainingData)
		{
			allInputData.addAll(dataPoint.getTargets());
		}
		return allInputData;
	}

    protected double[] unormalizeOutputs(double[] normalizedOutputs)
    {
        return unormalize(normalizedOutputs, minTarget, maxTarget) ;
    }

	// protected static double[][] unormalize(double[][] data, double min, double max)
	// {
	// 	double[][] unnormalizedData = new double[data.length][] ;
	// 	for (int i = 0 ; i <= data.length - 1 ; i += 1)
	// 	{
	// 		unnormalizedData[i] = new double[data[i].length] ;
	// 		for (int j = 0 ; j < data[i].length ; j++)
	// 		{
	// 			unnormalizedData[i][j] = unormalize(min, data[i][j], max) ;
	// 		}
	// 	}
	// 	return unnormalizedData ;
	// }

	protected static double[] unormalize(double[] data, double min, double max)
	{
		double[] unnormalizedData = new double[data.length] ;
		for (int i = 0 ; i <= data.length - 1 ; i += 1)
		{
            unnormalizedData[i] = unormalize(min, data[i], max) ;
			
		}
		return unnormalizedData ;
	}

	private static double unormalize(double min, double value, double max)
	{
		double newMin = MIN_NORMALIZED_VALUE ;
		double newMax = MAX_NORMALIZED_VALUE ;
		return (value - newMin) * (max - min) / (newMax - newMin) + min;
	}

	private static double normalize(double min, double value, double max)
	{
		double newMin = MIN_NORMALIZED_VALUE ;
		double newMax = MAX_NORMALIZED_VALUE ;
		return (value - min) * newMax / (max - min) + newMin ;
	}

    public List<DataPoint> getDataPoints() { return dataPoints ;}
    public List<Double> getListAllInputs() { return listAllInputs ;}
    public List<Double> getListAllTargets() { return listAllTargets ;}
    public List<DataPoint> getNormalizedDataPoints() { return normalizedDataPoints ;}
    public List<Double> getNormalizedListAllInputs() { return normalizedListAllInputs ;}
    public List<Double> getNormalizedListAllTargets() { return normalizedListAllTargets ;}
    public double getMinInput() { return minInput ;}
    public double getMaxInput() { return maxInput ;}
    public double getMinTarget() { return minTarget ;}
    public double getMaxTarget() { return maxTarget ;}

}
