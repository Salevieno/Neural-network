package neural.network;

import java.awt.Color;
import java.awt.Point;

import draw.Draw;
import graphics.Align;
import main.Main;

public class Neuron
{
    private static final int size = 20 ;
    private static final int stroke = 2 ;
    private static final Color color = Main.palette[1] ;

    public void display(Point pos, Align align)
    {
        Draw.DP.drawCircle(pos, size, stroke, color, color) ;
    }
}
