package main;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;


public class MainFrame extends JFrame
{
	private static final long serialVersionUID = 1L;
	private static final Dimension BUTTON_STD_SIZE = new Dimension(40, 40) ;
	private static final Dimension FRAME_SIZE = new Dimension(1100, 800) ;
	private static final MainPanel mainPanel = new MainPanel(new Dimension(FRAME_SIZE.width, FRAME_SIZE.height - BUTTON_STD_SIZE.height - 20)) ;

	private static final String ImagesPath ;
	private static final ImageIcon UseIcon ;
	private static final ImageIcon TestIcon ;
	private static final ImageIcon TrainIcon ;
	private static final ImageIcon ANNIcon ;
	private static final ImageIcon GraphsIcon ;

	private static final MainFrame instance ;
	

	static
	{
		ImagesPath = MainFrame.class.getResource("/assets/").getPath() ;
		UseIcon = new ImageIcon(ImagesPath + "PlayIcon.png");
		TestIcon = new ImageIcon(ImagesPath + "PlayIcon.png");
		TrainIcon = new ImageIcon(ImagesPath + "PlayIcon.png");
		ANNIcon = new ImageIcon(ImagesPath + "ANNIcon.png");
		GraphsIcon = new ImageIcon(ImagesPath + "graphs.png");
	
		instance = new MainFrame() ;
	}

	public static MainFrame getMe() { return instance ;}
	
	public void addButtons()
	{

		JButton trainOneIterationButton = createButton(UseIcon);
		JButton testButton = createButton(TestIcon);
		JButton trainButton = createButton(TrainIcon);
		JButton displayANNButton = createButton(ANNIcon);
		JButton displayGraphsButton = createButton(GraphsIcon);

		trainOneIterationButton.addActionListener(e -> mainPanel.trainOneIteration());
		testButton.addActionListener(e -> mainPanel.switchRunTesting());
		trainButton.addActionListener(e -> mainPanel.switchRunTraining());
		displayANNButton.addActionListener(e -> mainPanel.switchANNDisplay());
		displayGraphsButton.addActionListener(e -> mainPanel.switchGraphsDisplay());
		
		this.add(trainOneIterationButton);
		this.add(testButton);
		this.add(trainButton);
		this.add(displayANNButton);
		this.add(displayGraphsButton);
		
	}
	
	private static JButton createButton(ImageIcon icon, int alignX, int alignY, Dimension size, Color color)
	{
		JButton button = new JButton();
		button.setIcon(icon);
		button.setVerticalAlignment(alignX);
		button.setHorizontalAlignment(alignY);
		if (color != null)
		{			
			button.setBackground(color);
		}
		button.setPreferredSize(size);
		button.setFocusPainted(false);
		button.setBorderPainted(false);
		button.setContentAreaFilled(false);
		button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
		return button;
	}
	
	private static JButton createButton(ImageIcon icon)
	{
		return createButton(icon, 0, 0, BUTTON_STD_SIZE, Color.white) ;
	}

	public MainFrame()
	{
		this.setTitle("Rede neural");
		this.setSize(FRAME_SIZE);
		this.setLayout(new FlowLayout());
		this.addButtons();
		this.add(mainPanel);
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setVisible(true);
	}
}
