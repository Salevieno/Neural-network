package main;

import java.awt.Color;

public class Main
{
	public static final Color[] palette;
    
    static
    {        
		palette = new Color[] {
            new Color(63, 40, 231),
            new Color(137, 249, 204),
            new Color(11, 11, 11),
            new Color(225, 120, 170),
            new Color(241, 199, 128),
            new Color(101, 131, 246),
            new Color(76, 131, 42)
        } ;
    }

    public static void main(String[] args)
    {        
        MainFrame.getMe() ;
    }
}