package fiji.plugin.appose.yolo;

import java.awt.EventQueue;
import java.awt.Font;
import java.awt.Window;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.scijava.Initializable;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Overlay;
import ij.gui.Roi;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.appose.NDArrays;
import net.imglib2.appose.ShmImg;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;

import ij.ImageStack;
import ij.plugin.ChannelSplitter;
import ij.plugin.frame.RoiManager;
import ij.CompositeImage;

import javax.swing.JDialog;
import javax.swing.JProgressBar;
import javax.swing.WindowConstants;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;


/*
 * This class implements an example of a classical Fiji plugin (not ImageJ2 plugin), 
 * that calls native Python code with Appose.
 * 
 * We use a simple examples of rotating an input image by 90 degrees, using the scikit-image 
 * library in Python, and returning the result back to Fiji. Everything is contained in a 
 * single class, but you can imagine restructuring the code and the Python script as you see fit.
 */
@Plugin(type = Command.class, menuPath = "Plugins>Yolo-Appose")
public class YoloAppose extends DynamicCommand implements Initializable
{
	
	@Parameter
    private LogService log;
	
	@Parameter(label = "Model path", style = "file")
	private File ModelPath;
	
	@Parameter( label = "Confidence threshold", min="0.", description="Confidence threshold [0,1]" )
	private double ConfidenceThreshold = 0.5; // confidence threshold
	
	@Parameter( label = "Slice height", min="10", description="Slice height (in pixels)" )
	private int SliceHeight = 128; // slice height
	
	@Parameter( label = "Slice width", min="10", description="Slice width (in pixels)" )
	private int SliceWidth = 128; // slice height
	
	@Parameter( label = "Overlap height ratio", min="0.05", description="Overlap height ratio (in %)" )
	private double OverlapHeightRatio = 0.2; // slice height
	
	@Parameter( label = "Overlap width ratio", min="0.05", description="Overlap width ratio (in %)" )
	private double OverlapWidthRatio = 0.2; // slice height
	
	@Override
	public void initialize() {
		
	}
	
	/*
	 * This is the entry point for the plugin. This is what is called when the
	 * user select the plugin menu entry: 'Plugins>Yolo-Appose>Yolo appose' in our case. 
	 * You can redefine this by editing
	 * the file 'plugins.config' in the resources directory
	 * (src/main/resources).
	 */
	@Override
	public void run()
	{
		// Grab the current image.
		final ImagePlus imp = WindowManager.getCurrentImage();
		
		// Open image
//		ImagePlus imp = IJ.openImage(ImageFile.getAbsolutePath());
		
//		imp.show();
		
		try
		{
			// Runs the processing code.
			process( imp );
		}
		catch ( final IOException | BuildException e )
		{
			log.error( "An error occurred: " + e.getMessage() );
		}
	}

	/*
	 * Actually do something with the image.
	 */
	public < T extends RealType< T > & NativeType< T > > void process( final ImagePlus imp ) throws IOException, BuildException
	{
		// Print os and arch info
		System.out.println( "This machine os and arch:" );
		System.out.println( "  " + System.getProperty( "os.name" ) );
		System.out.println( "  " + System.getProperty( "os.arch" ) );
		System.out.println();
		
		/*
		 * Check the image type. If it is color RGB, then convert to composite image. 
		 * Appose doesn't support color RGB.
		 * TODO: update func to accept: 1 single channel, composite image with 3 channels and RGB image.
		 */
		
		final ImagePlus impPreprocessed = preprocess( imp );

		/*
		 * For this example we use pixi to create a Python environment with the
		 * necessary dependencies. We load it from an existing file
		 */

		// The environment spec.
		final String yoloEnv = loadResource( "/scripts/pixi.toml" );
		System.out.println( "The environment specs:" );
		System.out.println( indent( yoloEnv ) );
		System.out.println();

		/*
		 * The Python script that we want to run. It is loaded from an existing .py file. 
		 */

		// Get the script
		final String script = loadResource( "/scripts/yolo-sahi-detector.py" );
		System.out.println( "The analysis script" );
		System.out.println( indent( script ) );
		System.out.println();

		/*
		 * The following wraps an ImageJ ImagePlus into an ImgLib2 Img, and then
		 * into an Appose NDArray, which is a shared memory array that can be
		 * passed to Python without copying the data.
		 * 
		 * As an ImagePlus is not mapped on a shared memory array, the ImgLib2
		 * image wrapping the ImagePlus is actually copied to a shared memory
		 * image (the ShmImg) when we wrap it into an NDArray. This is because
		 * the NDArray needs to be backed by a shared memory array in order to
		 * be passed to Python without copying the data. We could have avoided
		 * this copy by directly loading the image into a ShmImg in the first
		 * place, but for simplicity we start with an ImagePlus and show how to
		 * wrap it into a shared memory array.
		 */

		// Wrap the ImagePlus into a ImgLib2 image.
		@SuppressWarnings( "unchecked" )
		final ImgPlus< T > img = rawWraps( impPreprocessed );

		/*
		 * Copy the image into a shared memory image and wrap it into an
		 * NDArray, then store it in an input map that we will pass to the
		 * Python script.
		 * 
		 * Note that we could have passed multiple inputs to the Python script
		 * by putting more entries in the input map, and they would all be
		 * available in the Python script as shared memory NDArrays.
		 * 
		 * A ND array is a multi-dimensional array that is stored in shared
		 * memory, that can be unwrapped as a NumPy array in Python, and wrapped
		 * as a ImgLib2 image in Java.
		 * 
		 */
		final Map< String, Object > inputs = new HashMap<>();
		
		try {
			inputs.put( "img_apos", NDArrays.asNDArray( img ) );
		}
		catch(IllegalArgumentException e) {
			log.info( e.toString() );
		}
		
//		final Integer SliceHeight = 128;
//		final Integer SliceWidth = 128;
//		final Double OverlapHeightRatio = 0.2;
//		final Double OverlapWidthRatio = 0.2;
		
		inputs.put( "model_path_apos", ModelPath.getAbsolutePath() );
		inputs.put( "confidence_threshold_apos", ConfidenceThreshold );
		inputs.put( "slice_height_apos", SliceHeight );
		inputs.put( "slice_width_apos", SliceWidth );
		inputs.put( "overlap_height_ratio_apos", OverlapHeightRatio );
		inputs.put( "overlap_width_ratio_apos", OverlapWidthRatio );
		
		/*
		 * Create or retrieve the environment.
		 * 
		 * The first time this code is run, Appose will create the mamba
		 * environment as specified by the cellposeEnv string, download and
		 * install the dependencies. This can take a few minutes, but it is only
		 * done once. The next time the code is run, Appose will just reuse the
		 * existing environment, so it will start much faster.
		 */
		final Environment env = Appose // the builder
				.pixi() // choose the environment manager
				.content( yoloEnv ) // specify the environment with the string defined above
				.subscribeProgress( this::showProgress ) // report progress visually
				.subscribeOutput( this::showProgress ) // report output visually
				.subscribeError( IJ::log ) // log problems
				.build(); // create the environment
//		hideProgress();

		/*
		 * Using this environment, we create a service that will run the Python
		 * script.
		 */
		try ( Service python = env.python() )
		{
			/*
			 * With this service, we can now create a task that will run the
			 * Python script with the specified inputs. This command takes the
			 * script as first argument, and a map of inputs as second argument.
			 * The keys of the map will be the variable names in the Python
			 * script, and the values are the data that will be passed to
			 * Python.
			 */
			final Task task = python.task( script, inputs );

			// Start the script, and return to Java immediately.
			log.info( "Starting Appose task." );
			final long start = System.currentTimeMillis();
			task.start();

			/*
			 * Wait for the script to finish. This will block the Java thread
			 * until the Python script is done, but it allows the Python code to
			 * run in parallel without blocking the Java thread while it is
			 * running.
			 */
			task.waitFor();

			// Verify that it worked.
			if ( task.status != TaskStatus.COMPLETE )
				throw new RuntimeException( "Python script failed with error: " + task.error );

			// Benchmark.
			final long end = System.currentTimeMillis();
			log.info( "Task finished in " + ( end - start ) / 1000. + " s" );

			/*
			 * Unwrap output.
			 * 
			 * In the Python script (see below), we create a new NDArray called
			 * 'rotated' that contains the result of the processing. Here we
			 * retrieve this NDArray from the task outputs, and wrap it into a
			 * ShmImg, which is an ImgLib2 image that is backed by shared
			 * memory. We can then display this image with
			 * ImageJFunctions.show(). Note that this does not involve any
			 * copying of the data, as the NDArray and the ShmImg are both just
			 * views on the same shared memory array.
			 */
//			final NDArray maskArr = ( NDArray ) task.outputs.get( "bboxlabel" );
//			final Img< T > output = new ShmImg<>( maskArr );
//			ImageJFunctions.show( output );
			
			/**
			 * Test getting bbox table
			 */
			@SuppressWarnings("unchecked")
            List<Map<String, Object>> bboxes = (List<Map<String, Object>>) task.outputs.get("bboxtable");
			
			if (bboxes == null || bboxes.isEmpty()) {
                IJ.showMessage("No detections found.");
                return;
            }
			
			// 6. Convert to Fiji ROIs
            RoiManager rm = RoiManager.getInstance();
            if (rm == null) rm = new RoiManager();

            Overlay overlay = new Overlay();

            for (Map<String, Object> bbox : bboxes) {
                double x      = ((Number) bbox.get("x")).doubleValue();
                double y      = ((Number) bbox.get("y")).doubleValue();
                double width  = ((Number) bbox.get("width")).doubleValue();
                double height = ((Number) bbox.get("height")).doubleValue();
                double conf   = ((Number) bbox.get("confidence")).doubleValue();
                int classId   = ((Number) bbox.get("class_id")).intValue();
                String name   = (String) bbox.get("class_name");

                // Create the rectangle ROI (the key conversion!)
                // Need to swap x and y to match convention between python and fiji
                Roi roi = new Roi(y, x, height, width);

                // Label and style
                String label = String.format("%s (%.2f)", name, conf);
                roi.setName(label);
//                roi.setStrokeColor(getColorForClass(classId));
                roi.setStrokeWidth(1);

                // Add to ROI Manager
                rm.addRoi(roi);

                // Also add to overlay for direct display
                overlay.add(roi);
            }

            // 7. Display results
            imp.setOverlay(overlay);
            imp.updateAndDraw();

            log.info("YOLO detection complete: " + bboxes.size() + " objects found.");
            
            /*
             * Display table
             */

            // Show results table
            DetectionTableDialog tableDialog = new DetectionTableDialog();
            tableDialog.show(bboxes);
		
		}
		catch ( Exception e )
		{
			// TODO Auto-generated catch block
//			e.printStackTrace();
			IJ.log( e.toString() );
		}
	}
	
	private ImagePlus preprocess(ImagePlus imp) {
		
	    if (imp.getType() != ImagePlus.COLOR_RGB) {
	    	
	        return imp;
	    }

	    log.info( "Image is RGB, convert to composite stack." );
	    
	    ImagePlus[] channels = ChannelSplitter.split(imp);

	    ImageStack stack = new ImageStack(imp.getWidth(), imp.getHeight());
	    stack.addSlice("Red", channels[0].getProcessor());
	    stack.addSlice("Green", channels[1].getProcessor());
	    stack.addSlice("Blue", channels[2].getProcessor());

	    ImagePlus result = new ImagePlus(imp.getTitle(), stack);
	    result.setDimensions(3, 1, 1);
	    CompositeImage composite = new CompositeImage(result, CompositeImage.COMPOSITE);
	    composite.setCalibration(imp.getCalibration()); // preserve spatial calibration

	    return composite;
	}
	
    private String loadResource(String path) throws IOException {
        InputStream in = getClass().getResourceAsStream(path);
        if (in == null) {
            throw new FileNotFoundException("Resource not found: " + path);
        }
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            return sb.toString();
        }
    }

	// Helper functions to display progress while building the Appose environment.
	// Temporary solution until Appose has a nicer built-in way to do this.

	private JDialog progressDialog;
	private JProgressBar progressBar;
	private void showProgress( String msg )
	{
		showProgress( msg, null, null );
	}
	private void showProgress( String msg, Long cur, Long max )
	{
		EventQueue.invokeLater( () ->
		{
			if ( progressDialog == null ) {
				Window owner = IJ.getInstance();
				progressDialog = new JDialog( owner, "Fiji ♥ Appose" );
				progressDialog.setDefaultCloseOperation( WindowConstants.DO_NOTHING_ON_CLOSE );
				progressBar = new JProgressBar();
				progressDialog.getContentPane().add( progressBar );
				progressBar.setFont( new Font( "Courier", Font.PLAIN, 14 ) );
				progressBar.setString(
					"--------------------==================== " +
					"Building Python environment " +
					"====================--------------------"
				);
				progressBar.setStringPainted( true );
				progressBar.setIndeterminate( true );
				progressDialog.pack();
				progressDialog.setLocationRelativeTo( owner );
				progressDialog.setVisible( true );
			}
			if ( msg != null && !msg.trim().isEmpty() ) progressBar.setString( "Building Python environment: " + msg.trim() );
			if ( cur != null || max != null ) progressBar.setIndeterminate( false );
			if ( max != null ) progressBar.setMaximum( max.intValue() );
			if ( cur != null ) progressBar.setValue( cur.intValue() );
		} );
	}
//	private void hideProgress()
//	{
//		EventQueue.invokeLater( () ->
//		{
//			progressDialog.dispose();
//			progressDialog = null;
//		} );
//	}

	/*
	 * A utility to pretty print things. Probably will go away in your code.
	 */
	private static String indent( final String script )
	{
		final String[] split = script.split( "\n" );
		String out = "";
		for ( final String string : split )
			out += "    " + string + "\n";
		return out;
	}

	/*
	 * A utility to wrap an ImagePlus into an ImgPlus, without too many
	 * warnings. Hacky.
	 */
	@SuppressWarnings( "rawtypes" )
	public static final ImgPlus rawWraps( final ImagePlus imp )
	{
		final ImgPlus< DoubleType > img = ImagePlusAdapter.wrapImgPlus( imp );
		final ImgPlus raw = img;
		return raw;
	}

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