package draw;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Point;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

import graphics.Align;
import graphics.DrawPrimitives;
import graphics.UtilAlignment;
import main.Main;
import utilities.Util;

public abstract class Draw
{
	private static Font smallFont = new Font("SansSerif", Font.BOLD, 13);	
	public static final DrawPrimitives DP = new DrawPrimitives();
	
	
	public static void setGraphics(Graphics2D g) { DP.setGraphics((Graphics2D) g) ;}		

	public static void text(Point pos, String text, Color color)
	{
		text(pos, Align.topLeft, text, smallFont, color);
	}
	
	public static void text(Point pos, Align align, String text, Color color)
	{
		text(pos, align, text, smallFont, color);
	}
	
	public static void text(Point pos, Align align, String text, Font font, Color color)
	{
		text(pos, align, 0, text, font, color);
	}
	
	public static void text(Point pos, Align align, double angle, String text, Font font, Color color)
	{
		DP.drawText(pos, align, angle, text, font, color);
	}
	
    public static void menu(Point pos, Align align, int l, int h, int Thickness, Color[] colors, Color ContourColor)
    {
    	int border = 3;
		Dimension size = new Dimension(l, h);
		Point topLeftBorder = UtilAlignment.getPosAt(pos, align, Align.topLeft, size) ;
		topLeftBorder = Util.translate(topLeftBorder, -border, -border) ;

    	// DP.drawRect(topLeftBorder, Align.topLeft, size, Thickness, colors[0], ContourColor, 1.0) ;
    	DP.drawRect(pos, align, new Dimension(l - 2*border, h - 2*border), Thickness, colors[1], Main.palette[0], 1.0) ;
    }
    public static void menu(Point pos, Align align, int l, int h)
    {
    	menu(pos, align, l, h, 1, new Color[] { Main.palette[0], Main.palette[4] }, Main.palette[0]) ;
    }
    public static void menu(Point pos, Align align)
    {
    	menu(pos, align, 200, 200, 2, new Color[] { Main.palette[6], Main.palette[3] }, Main.palette[2]) ;
    }
    
	private static void neurons(Point pos, Dimension size, int[] Nneurons)
	{
		int NeuronSize = 36 ;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		Color fillColor = Main.palette[1] ;

		for (int l = 0; l <= Nneurons.length - 1; l += 1)
		{
			int sy = (size.height - NeuronSize * Nneurons[l]) / (Nneurons[l] + 1);
			for (int n = 0; n <= Nneurons[l] - 1; n += 1)
			{
				Point NeuronPos = new Point(pos.x + l * (sx + NeuronSize) + sx + NeuronSize / 2, pos.y + n * (sy + NeuronSize) + sy + NeuronSize / 2) ;
				DP.drawCircle(NeuronPos, NeuronSize, 2, fillColor, Main.palette[2]) ;
			}
		}
	}
	
	private static void neuronValues(Point pos, Dimension size, int[] Nneurons, double[][] neuronvalue)
	{
		int NeuronSize = 30;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		Color color = Main.palette[2] ;

		for (int l = 0; l <= Nneurons.length - 1; l += 1)
		{
			int sy = (size.height - NeuronSize * Nneurons[l]) / (Nneurons[l] + 1);
			for (int n = 0; n <= Nneurons[l] - 1; n += 1)
			{
				Point NeuronPos = new Point(pos.x + l * (sx + NeuronSize) + sx + NeuronSize / 2, pos.y + n * (sy + NeuronSize) + sy + NeuronSize / 2) ;
				String neuronValue = Double.isFinite(neuronvalue[l][n]) ? String.valueOf(round(neuronvalue[l][n], 2)) : "∞" ;
				DP.drawText(NeuronPos, Align.center, 0, neuronValue, smallFont, color);
			}
		}
	}

	private static void biasValues(Point pos, Dimension size, int[] Nneurons, double[][] biasValue)
	{
		int NeuronSize = 30;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		Color color = Main.palette[6] ;

		for (int l = 0; l <= Nneurons.length - 1; l += 1)
		{
			int sy = (size.height - NeuronSize * Nneurons[l]) / (Nneurons[l] + 1);
			for (int n = 0; n <= Nneurons[l] - 1; n += 1)
			{
				Point textPos = new Point(pos.x + l * (sx + NeuronSize) + sx + NeuronSize / 2, pos.y + n * (sy + NeuronSize) + sy + NeuronSize / 2 - 25) ;
				String biasValueText = Double.isFinite(biasValue[l][n]) ? String.valueOf(round(biasValue[l][n], 2)) : "∞" ;
				DP.drawText(textPos, Align.center, 0, "(" + biasValueText + ")", smallFont, color);
			}
		}
	}

	private static void weightValues(Point pos, Dimension size, int[] Nneurons, double[][][] weightValue)
	{
		int NeuronSize = 30;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		Color color = Main.palette[5] ;

		for (int l = 0; l <= Nneurons.length - 2; l += 1)
		{
			int sy = (size.height - NeuronSize * Nneurons[l]) / (Nneurons[l] + 1);
			for (int n1 = 0; n1 <= Nneurons[l + 1] - 1; n1 += 1)
			{
				for (int n2 = 0; n2 <= Nneurons[l] - 1; n2 += 1)
				{
					Point textPos = new Point(pos.x + l * (sx + NeuronSize) + 3 * sx / 2 + NeuronSize / 2, pos.y + n1 * (sy + NeuronSize) / 2 + n2 * (sy / 2 + NeuronSize) + 3 * sy / 2 + NeuronSize / 2 - 20) ;
					String biasValueText = Double.isFinite(weightValue[l][n1][n2]) ? String.valueOf(round(weightValue[l][n1][n2], 2)) : "∞" ;
					DP.drawText(textPos, Align.center, 0, biasValueText, smallFont, color);
				}
			}
		}
	}

	private static void inputValues(Point pos, Dimension size, int[] Nneurons, List<Double> inputs)
	{
		int NeuronSize = 30;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		int sy = (size.height - NeuronSize * Nneurons[0]) / (Nneurons[0] + 1);
		for (int i = 0 ; i <= Nneurons[0] - 1 ; i += 1)
		{
			Point NeuronPos = new Point(pos.x + sx / 2, pos.y + i * (sy + NeuronSize) + sy + NeuronSize / 2) ;
			DP.drawText(NeuronPos, Align.center, 0, String.valueOf(round(inputs.get(i), 2)), smallFont, Main.palette[2]) ;
		}
	}

	private static void outputValues(Point pos, Dimension size, int[] Nneurons, List<Double> targets)
	{
		int NeuronSize = 30;
		int sy = (size.height - NeuronSize * Nneurons[Nneurons.length - 1]) / (Nneurons[Nneurons.length - 1] + 1);
		for (int i = 0 ; i <= Nneurons[Nneurons.length - 1] - 1 ; i += 1)
		{
			Point targetTextPos = new Point(pos.x + size.width - 30, pos.y + i * (sy + NeuronSize) + sy + NeuronSize / 2) ;
			DP.drawText(targetTextPos, Align.center, 0, String.valueOf(round(targets.get(i), 2)), smallFont, Main.palette[2]) ;
		}
	}

	private static void connectionLines(Point pos, Dimension size, int[] Nneurons, double[][][] weight, double MaxWeight)
	{
		int NeuronSize = 30;
		int sx = (size.width - NeuronSize * Nneurons.length) / (Nneurons.length + 1);
		for (int l = 1; l <= Nneurons.length - 1;l += 1)
		{
			int sy1 = (size.height - NeuronSize * Nneurons[l - 1]) / (Nneurons[l - 1] + 1);
			int sy2 = (size.height - NeuronSize * Nneurons[l]) / (Nneurons[l] + 1);
			for (int n1 = 0; n1 <= Nneurons[l - 1] - 1; n1 += 1)
			{
				for (int n2 = 0; n2 <= Nneurons[l] - 1; n2 += 1)
				{
					Point NeuronPos1 = new Point(pos.x + (l - 1) * (sx + NeuronSize) + sx + NeuronSize / 2, pos.y + n1 * (sy1 + NeuronSize) + sy1 + NeuronSize / 2) ;
					Point NeuronPos2 = new Point(pos.x + l * (sx + NeuronSize) + sx + NeuronSize / 2, pos.y + n2 * (sy2 + NeuronSize) + sy2 + NeuronSize / 2) ;
					
					int alpha = (int) (255 * Math.abs(weight[l - 1][n2][n1]) / MaxWeight) ;
					Color lineColor = 0 < weight[l - 1][n2][n1] ? new Color(0, 180, 0, alpha) : new Color(200, 0, 0, alpha) ;
					DP.drawLine(NeuronPos1, NeuronPos2, 2, lineColor) ;
				}
			}
		}
	}

	public static void ann(Point pos, Dimension size, int[] Nneurons, List<Double> inputs, List<Double> targets, double[][] neuronvalue, double[][][] weight, double MaxWeight, boolean DrawLines, Color color, double[][] biases)
	{
		if (DrawLines)
		{
			connectionLines(pos, size, Nneurons, weight, MaxWeight) ;
		}
		
		neurons(pos, size, Nneurons) ;
		neuronValues(pos, size, Nneurons, neuronvalue) ;
		biasValues(pos, size, Nneurons, biases) ;
		weightValues(pos, size, Nneurons, weight) ;
		if (inputs != null)
		{
			inputValues(pos, size, Nneurons, inputs) ;
		}
		if (targets != null)
		{
			outputValues(pos, size, Nneurons, targets) ;
		}
	}

	public static void ann(Point pos, int[] Nneurons, List<Double> inputs, List<Double> targets, double[][] neuronvalue, double[][][] weight, double MaxWeight, double[][] biases)
	{
		ann(pos, new Dimension(500, 200), Nneurons, inputs, targets, neuronvalue, weight, MaxWeight, true, Main.palette[0], biases) ;
	}
	
	private static float round(double number, int decimals) { return BigDecimal.valueOf(number).setScale(decimals, RoundingMode.HALF_EVEN).floatValue() ;}

}
