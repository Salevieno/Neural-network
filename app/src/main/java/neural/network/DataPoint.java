package neural.network;

import java.util.ArrayList;
import java.util.List;

public class DataPoint
{    

	private List<Double> inputs ;
	private List<Double> targets ;

	public DataPoint(List<Double> inputData, List<Double> targetData)
	{
		this.inputs = inputData ;
		this.targets = targetData ;
	}

	public DataPoint()
	{
		this(new ArrayList<>(), new ArrayList<>()) ;
	}

    public List<Double> getInputs() { return inputs ;}
    public List<Double> getTargets() { return targets ;}

	@Override
	public String toString()
	{
		return "DataPoint [inputs=" + inputs + ", targets=" + targets + "]";
	}
	
}
