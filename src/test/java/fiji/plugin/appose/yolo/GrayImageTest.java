package fiji.plugin.appose.yolo;

import ij.IJ;
import net.imagej.ImageJ;

public class GrayImageTest
{

	public static void main( final String[] args )
	{
		// Launch ImageJ2
	    final ImageJ ij = new ImageJ();
	    ij.ui().showUI();

	    // Open an image BEFORE running the plugin
	    IJ.openImage("samples/celegan/I_tm8651.tif").show();

	    // Run the plugin via ImageJ2 context (NOT new My_Plugin().run())
	    ij.command().run(YoloAppose.class, true);
		
	}
}
