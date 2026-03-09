package fiji.plugin.appose.yolo;

import ij.IJ;
import net.imagej.ImageJ;

public class RGBImageTest
{
	public static void main( final String[] args )
	{
		// Launch ImageJ2
	    final ImageJ ij = new ImageJ();
	    ij.ui().showUI();

	    // Open an image BEFORE running the plugin
	    IJ.openImage("samples/sporozoite/sporozoite.tif").show();

	    // Run the plugin via ImageJ2 context (NOT new My_Plugin().run())
	    ij.command().run(YoloAppose.class, true);
		
	}

}
