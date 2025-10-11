package neural.network;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
import java.math.BigDecimal;
import java.math.RoundingMode;

import draw.Draw;
import graphics.Align;
import main.Palette;

public class InfoPanel extends Draggable
{
    public InfoPanel(Point topLeftPos)
    {
        this.topLeftPos = topLeftPos ;
        this.size = new Dimension(120, 100) ;
    }


	private static float round(double number, int decimals) { return BigDecimal.valueOf(number).setScale(decimals, RoundingMode.HALF_EVEN).floatValue() ;}

    public void display(boolean biasIsActive, int iter, double learningRate, double errorPerc, Mode mode)
	{
		Draw.menu(topLeftPos, Align.topLeft, size.width, size.height, 2, new Color[] { Palette.blue, Palette.cyan }, null);

		int sy = 15;
		Color textColor = Color.black ;
		Point contentPos = new Point(topLeftPos.x + 5, topLeftPos.y + 5) ;
		String[] lineContent =
		{
			"General info",
			"Bias: " + String.valueOf(biasIsActive),
			"Iteração: " + String.valueOf(iter),
			"Lrate: " + String.valueOf(learningRate),
			"Erro: " + String.valueOf(round(errorPerc, 2)),
			"Mode: " + mode
		} ;
		for (int i = 0 ; i <= lineContent.length - 1 ; i += 1)
		{			
			Draw.text(contentPos, lineContent[i], textColor);
			contentPos.y += sy ;
		}
	}
}
