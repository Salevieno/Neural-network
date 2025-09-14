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
	private static final Dimension frameSize = new Dimension(700, 700) ;
	private static final MainPanel mainPanel = new MainPanel(new Dimension(700, 600));

	private static final String ImagesPath ;
	private static final ImageIcon UseIcon ;
	private static final ImageIcon PlayIcon ;
	private static final ImageIcon ANNIcon ;
	private static final ImageIcon GraphsIcon ;

	private static final MainFrame instance ;
	

	static
	{
		ImagesPath = MainFrame.class.getResource("/assets/").getPath() ;
		UseIcon = new ImageIcon(ImagesPath + "PlayIcon.png");
		PlayIcon = new ImageIcon(ImagesPath + "PlayIcon.png");
		ANNIcon = new ImageIcon(ImagesPath + "ANNIcon.png");
		GraphsIcon = new ImageIcon(ImagesPath + "graphs.png");
	
		instance = new MainFrame() ;
	}

	public static MainFrame getMe() { return instance ;}
	
	public void addButtons()
	{

		JButton UseButton = createButton(UseIcon);
		JButton PlayButton = createButton(PlayIcon);
		JButton NNButton = createButton(ANNIcon);
		JButton GraphsButton = createButton(GraphsIcon);

		UseButton.addActionListener(e -> mainPanel.useNetworks());
		PlayButton.addActionListener(e -> mainPanel.switchRunTraining());
		NNButton.addActionListener(e -> mainPanel.switchANNDisplay());
		GraphsButton.addActionListener(e -> mainPanel.switchGraphsDisplay());
		
		this.add(UseButton);
		this.add(PlayButton);
		this.add(NNButton);
		this.add(GraphsButton);
		
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
		this.setSize(frameSize);
		this.setLayout(new FlowLayout());
		this.addButtons();
		this.add(mainPanel);
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setVisible(true);
	}
}
