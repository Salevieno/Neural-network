package neural.network;

import java.awt.Dimension;
import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

public abstract class Draggable
{
    protected Point topLeftPos ;
    protected Dimension size ;
    protected boolean wasClicked = false ;

    private static final List<Draggable> all = new ArrayList<>() ;

    public Draggable()
    {
        all.add(this) ;
    }

    public static List<Draggable> getAll() { return all ;}

    public void incTopLeftPos(Point delta) { this.topLeftPos.translate(delta.x, delta.y) ;}

    public void updateWasClicked(Point mousePos) {  wasClicked = mouseIsInside(mousePos) ; }

    public void resetWasClicked() { wasClicked = false ;}

    public boolean mouseIsInside(Point mouse)
    {
        return (mouse.x >= topLeftPos.x && mouse.x <= topLeftPos.x + size.width) &&
               (mouse.y >= topLeftPos.y && mouse.y <= topLeftPos.y + size.height) ;
    }

    public boolean wasClicked() { return wasClicked ;}
}
