package main;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Point;
import java.util.List;

import draw.Draw;
import graphics.Align;

public class Palette
{
    public static final Color blue = new Color(63, 40, 231);
    public static final Color cyan = new Color(137, 249, 204);
    public static final Color black = new Color(11, 11, 11);
    public static final Color pink = new Color(225, 120, 170);
    public static final Color orange = new Color(241, 199, 128);
    public static final Color purple = new Color(101, 131, 246);
    public static final Color green = new Color(76, 131, 42);

    private static List<Color> allColors() { return List.of(blue, cyan, black, pink, orange, purple, green) ;}

    public static void display()
    {
        int nCol = 4 ;
        Dimension size = new Dimension(30, 30) ;

        for (int i = 0 ; i <= allColors().size() - 1 ; i += 1)
        {
            Point topLeft = new Point(i % nCol * size.width, i / nCol * size.height) ;
            Draw.DP.drawRect(topLeft, Align.topLeft, size, allColors().get(i), null) ;
        }
    }
}
