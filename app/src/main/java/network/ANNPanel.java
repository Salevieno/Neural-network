package network;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import draw.Draw;
import graphics.Align;
import main.Palette;

public class ANNPanel extends Draggable
{

    public ANNPanel(Point topLeftPos)
    {
        this.topLeftPos = topLeftPos ;
        this.size = new Dimension(500, 200) ;
    }
	
	public void display(int[] qtdNeuronsInLayer, List<Double> unormalizedInputs, List<Double> unormalizedTargets, double[][] neuronvalue, double[][][] weights, double maxWeight)
	{
		
		Draw.menu(topLeftPos, Align.topLeft, size.width, size.height, 2, new Color[] { Palette.blue, Palette.orange }, null) ;
		Draw.ann(new Point(topLeftPos.x , topLeftPos.y), qtdNeuronsInLayer, unormalizedInputs, unormalizedTargets, neuronvalue, weights, maxWeight) ;

	}
}
