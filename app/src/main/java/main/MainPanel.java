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
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;
import javax.swing.Timer;

import org.ejml.simple.SimpleMatrix;

import draw.Draw;
import network.ANNMatricialVisual;
import network.Data;
import network.Draggable;
import network.Mode;

public class MainPanel extends JPanel implements ActionListener, MouseListener, MouseMotionListener 
{
	private static final long serialVersionUID = 1L;
	private final Timer timer;
	private final Color bgColor = new Color(10, 20, 50) ;
	private final Data trainingData = new Data("training_data.json") ;
	private final Data testingData = new Data("testing_data.json") ;
	private final ANNMatricialVisual ann3 ;

	private boolean trainingIsRunning = false ;
	private boolean testingIsRunning = false ;
	private boolean showANN = true ;
	private boolean showGraphs = false;

	private Point prevMousePos ;

	public MainPanel(Dimension size)
	{
		this.setBackground(bgColor);
		this.setPreferredSize(size);
		this.setVisible(true);	
		this.addMouseListener(this);
        this.addMouseMotionListener(this);

		ann3 = new ANNMatricialVisual(new Point(40, 500), new int[] {2, 1}, true, false, true, true) ;
		ann3.activateBiases() ;
		System.out.println(ann3.getBiases().get(0));


		List<SimpleMatrix> biases = new ArrayList<>() ;
		biases.add(new SimpleMatrix(new double[][] {{-0.6}, {-0.8}})) ;
		biases.add(new SimpleMatrix(new double[][] {{0.0}})) ;


		List<SimpleMatrix> weights = new ArrayList<>() ;
		weights.add(new SimpleMatrix(new double[][] {{50, 30}})) ;


		ann3.setBiases(biases);
		ann3.setWeights(weights);
		
		timer = new Timer(0, this);
		timer.start();
	}
	
	public void trainOneIteration()
	{
		ann3.train(trainingData.getDataPoints());
		if (ann3 instanceof ANNMatricialVisual)
		{
			((ANNMatricialVisual) ann3).updateResults(trainingData.getDataPoints()) ;
		}
	}
	public void switchRunTraining()
	{
		ann3.setMode(Mode.train) ;
		trainingIsRunning = !trainingIsRunning ;
		testingIsRunning = false ;
	}
	public void switchRunTesting()
	{
		ann3.setMode(Mode.test) ;
		trainingIsRunning = false ;
		testingIsRunning = !testingIsRunning ;
	}
	public void switchANNDisplay() { showANN = !showANN ;}
	public void switchGraphsDisplay() { showGraphs = !showGraphs ;}
	
	private void update()
	{
		run() ;
		display();
	}

	private void run()
	{
		if (!trainingIsRunning && !testingIsRunning) { return ;}

		ann3.run(trainingData.getDataPoints(), testingData.getDataPoints()) ;
		testingIsRunning = false ;
	}
	
	private void display()
	{
		if (!showANN) { return ;}

		ann3.displayInfoPanel() ;
		ann3.display() ;
		ann3.displayTrainingResultGraph() ;
		ann3.displayErrorGraph() ;
	}

	@Override
	public void paintComponent(Graphics g)
	{
		super.paintComponent(g);
		Draw.setGraphics((Graphics2D) g);
		update();
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
			ann3.printState() ;
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
