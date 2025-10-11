package main;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.JPanel;
import javax.swing.Timer;

import draw.Draw;
import neural.network.ANN1;
import neural.network.ANN2;
import neural.network.Data;
import neural.network.Draggable;

public class MainPanel extends JPanel implements ActionListener, MouseListener, MouseMotionListener 
{
	private static final long serialVersionUID = 1L;
	private final Timer timer;
	private final Color bgColor = new Color(10, 20, 50) ;
	// private final Dataset dataset = new Dataset() ;
	// private final Dataset dataset2 = new Dataset() ;
	// private final Dataset dataset3 = new Dataset() ;
	// private final Chart graph = new Chart(new Point(60, 580), "Posição x", 100) ;
	// private final Chart graph2 = new Chart(new Point(260, 580), "Posição y", 100) ;
	// private final Chart graph3 = new Chart(new Point(460, 580), "Inputs", 100) ;

	private final ANN1 ann1 ;
	private final ANN2 ann2 ;
	protected final Data trainingData = new Data("input.json") ;

	private boolean RunTraining = false ;
	private boolean ShowANN = true ;
	private boolean ShowGraphs = false;

	private Point prevMousePos ;

	public MainPanel(Dimension size)
	{
		this.setBackground(bgColor);
		this.setPreferredSize(size);
		this.setVisible(true);	
		this.addMouseListener(this);
        this.addMouseMotionListener(this);

		ann1 = new ANN1(false) ;
		ann2 = new ANN2(false, false) ;		

		// graph.addDataset(dataset);
		// graph.setSize(150) ;
		// graph.setGridColor(new Color(0, 0, 0, 60)) ;
		// graph.setDataSetColor(List.of(Color.blue)) ;
		// graph.setDataSetContourColor(List.of(Color.cyan)) ;
		
		// graph2.addDataset(dataset2);
		// graph2.setSize(150) ;
		// graph2.setGridColor(new Color(0, 0, 0, 60)) ;
		// graph2.setDataSetColor(List.of(Color.orange)) ;
		// graph2.setDataSetContourColor(List.of(Color.red)) ;

		// graph3.setSize(150) ;
		// graph3.setGridColor(new Color(0, 0, 0, 60)) ;
		// graph3.setDataSetColor(List.of(new Color(0, 180, 60))) ;
		// graph3.setDataSetContourColor(List.of(new Color(0, 90, 60))) ;
		
		timer = new Timer(0, this);
		timer.start();
	}
	
	public void trainOneIteration()
	{
		// List<Double> inputs = List.of(0.2, 0.2) ;
		// ann1.use(inputs) ;
		// ann1.updateResults() ;
		// ann2.use(inputs) ;
		// ann2.updateResults() ;
		ann2.train(trainingData.getDataPoints());
		ann2.updateResults();
	}
	public void switchRunTraining() { RunTraining = !RunTraining ;}
	public void switchANNDisplay() { ShowANN = !ShowANN ;}
	public void switchGraphsDisplay() { ShowGraphs = !ShowGraphs ;}
	
	private void run()
	{
		// ann1.run(trainingData.getDataPoints()) ;
		// ann1.updateResults() ;

		// double[] targets = Util.Transpose(ann2.getTarget())[0];
		// double[] outputs = Util.Transpose(ann2.getOutput())[0];
		// dataset.setX(trainingData.getDataPoints().get(0).getTargets()) ;
		// dataset.setY(ann2.getOutputsAsList()) ;
		// graph.updateDataset(dataset) ;

		ann2.run(trainingData.getDataPoints()) ;
		ann2.updateResults() ;
	}
	
	public void display()
	{
		ann1.displayInfoPanel() ;
		ann2.displayInfoPanel() ;
		if (ShowANN)
		{
			ann1.display();
			ann2.display() ;
			ann2.displayTrainingResultGraph();
		}

		// Palette.display() ;
	}

	@Override
	public void paintComponent(Graphics g)
	{
		super.paintComponent(g);
		Draw.setGraphics((Graphics2D) g);
		if (RunTraining)
		{
			run() ;
		}
		display();
	}

	@Override
	public void actionPerformed(ActionEvent e)
	{
		if (e.getSource() == timer)
		{// TODO repaiting efficiency
			repaint();
		}
	}

	@Override
	public void mouseDragged(MouseEvent e)
	{
		Point mousePos = new Point(e.getX(), e.getY()) ;		
		Point deltaMousePos = new Point(mousePos.x - prevMousePos.x, mousePos.y - prevMousePos.y) ;

		Draggable.getAll().stream().filter(Draggable::wasClicked).forEach(draggable -> draggable.incTopLeftPos(deltaMousePos)) ;

		prevMousePos = new Point(mousePos) ;
	}

	@Override
	public void mouseReleased(MouseEvent e)
	{
		Draggable.getAll().forEach(Draggable::resetWasClicked) ;
	}

	@Override
	public void mouseEntered(MouseEvent e)
	{
		
	}

	@Override
	public void mouseExited(MouseEvent e)
	{
		
	}

	@Override
	public void mouseMoved(MouseEvent e)
	{
		
	}

	@Override
	public void mouseClicked(MouseEvent e)
	{
		// print ANN2 state if right click
		if (e.getButton() == MouseEvent.BUTTON3)
		{
			ann2.printState() ;
		}
	}

	@Override
	public void mousePressed(MouseEvent e)
	{
		Point mousePos = new Point(e.getX(), e.getY()) ;
		Draggable.getAll().forEach(draggable -> draggable.updateWasClicked(mousePos)) ;
		prevMousePos = new Point(mousePos) ;
	}
}
