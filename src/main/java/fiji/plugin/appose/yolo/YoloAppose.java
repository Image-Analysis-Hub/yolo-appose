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


/*
 * This class implements Fiji plugin that calls native Python code with Appose.
 * The python code run YOLO from inputs given the Fiji plugin.
 */
@Plugin(type = Command.class, menuPath = "Plugins>Yolo-Appose")
public class YoloAppose extends DynamicCommand implements Initializable
{
	
	@Parameter
    private LogService log;
	
	@Parameter(label = "Model path", style = "file")
	private File modelPath; // path to the model
	
	@Parameter( label = "Confidence threshold", min="0.", description="Confidence threshold [0,1]" )
	private double confidenceThreshold = 0.5; // confidence threshold
	
	@Parameter( label = "Slice height", min="10", description="Slice height (in pixels)" )
	private int sliceHeight = 128; // slice height
	
	@Parameter( label = "Slice width", min="10", description="Slice width (in pixels)" )
	private int sliceWidth = 128; // slice height
	
	@Parameter( label = "Overlap height ratio", min="0.05", description="Overlap height ratio (in %)" )
	private double overlapHeightRatio = 0.2; // slice height
	
	@Parameter( label = "Overlap width ratio", min="0.05", description="Overlap width ratio (in %)" )
	private double overlapWidthRatio = 0.2; // slice height
	
	@Override
	public void initialize() {
		
	}
	
	/*
	 * This is the entry point for the plugin. This is what is called when the
	 * user selects the plugin menu entry: 'Plugins>Yolo-Appose>Yolo appose'. 
	 * You can redefine this by editing the file 'plugins.config' in the resources directory
	 * (src/main/resources).
	 */
	@Override
	public void run()
	{
		// Grab the current image.
		final ImagePlus imp = WindowManager.getCurrentImage();
		
		try
		{
			// Runs the processing code.
			process( imp );
		}
		catch ( final IOException | BuildException e )
		{
			IJ.error( "An error occurred: " + e.getMessage() );
		}
	}

	/*
	 * Principle function to process image.
	 */
	public < T extends RealType< T > & NativeType< T > > void process( final ImagePlus imp ) throws IOException, BuildException
	{
		// Print os and arch info
		IJ.log( "OS:  " + System.getProperty( "os.name" ) );
		IJ.log( "Arch:  " + System.getProperty( "os.arch" ) );
		
		/*
		 * Check the image type. If it is colorRGB, then convert to composite 3 channels image.
		 * Since Appose does not support colorRGB 24 bit.
		 */
		
		final ImagePlus impPreprocessed = preprocess( imp );

		/*
		 * Create Python environment with Pixi. We load it from an existing .toml file.
		 */
		String yoloEnv = null;
		try{
			IJ.log( "Loading python environment." );
			yoloEnv = loadResource( "/scripts/pixi.toml" );
		}
		catch(IOException e) {
			IJ.error( "Failed to load environment specification: " + e.getMessage() );
			return;
		}

		/*
		 * Python script that we want to run the service. It is loaded from an existing .py file. 
		 */
		
		String script = null;
		try{
			IJ.log("Loading python script.");
			script = loadResource( "/scripts/yolo-sahi-detector.py" );
		}
		catch(IOException e) {
			IJ.error( "Failed to load Python script: " + e.getMessage() );
			return;
		}
			

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
		@SuppressWarnings( "unchecked")
		final ImgPlus< T > img = rawWraps( impPreprocessed );

		/*
		 * Copy the image into a shared memory image and wrap it into an
		 * NDArray, then store it in an input map that we will pass to the
		 * Python script.
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
			IJ.error( e.toString() );
		}
		

		/*
		 * Put other parameters to inputs
		 */
		
		inputs.put( "model_path_apos", modelPath.getAbsolutePath() );
		inputs.put( "confidence_threshold_apos", confidenceThreshold );
		inputs.put( "slice_height_apos", sliceHeight );
		inputs.put( "slice_width_apos", sliceWidth );
		inputs.put( "overlap_height_ratio_apos", overlapHeightRatio );
		inputs.put( "overlap_width_ratio_apos", overlapWidthRatio );
		
		/*
		 * Create or retrieve the environment.
		 * 
		 * The first time this code is run, Appose will create the 
		 * environment as specified by the yoloEnv string, download and
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

			/*
			 * Listen for updates from Python side.
			 */
	        task.listen(event -> {
	            switch (event.responseType) {

	                case UPDATE:
	                    IJ.log("[Python backend]" + event.message);
	                    break;

	                case COMPLETION:
	                    IJ.log("Task completed successfully.");
	                    break;

	                case CANCELATION:
	                    IJ.log("Task was cancelled.");
	                    break;

	                case FAILURE:
	                    IJ.error("Task failed", event.task.error);
	                    break;
	            }
	        });
	        
	        // Start the script, and return to Java immediately.
 			IJ.log( "Starting Appose task." );
	     			
 			final long start = System.currentTimeMillis();
 			task.start();
	        
			/*
			 * Wait for the script to finish. This will block the Java thread
			 * until the Python script is done, but it allows the Python code to
			 * run in parallel without blocking the Java thread while it is
			 * running.
			 */
			task.waitFor();

			// Benchmark.
			final long end = System.currentTimeMillis();
			IJ.log( "Task finished in " + ( end - start ) / 1000. + " s" );

			/*
			 * Unwrap output.
			 * The output from python is a list of dictionnary, each item is bbox information.
			 */
			
			/**
			 * Test getting bbox table
			 */
			@SuppressWarnings("unchecked")
            List<Map<String, Object>> bboxes = (List<Map<String, Object>>) task.outputs.get("bboxtable");
			
			if (bboxes == null || bboxes.isEmpty()) {
                IJ.log("No detections found.");
                return;
            }
			
//			IJ.log("There are " + bboxes.size() + " objects found.");
			
			/*
			 * Convert to bbox coordinates to ROIs and put into RoiManager.
			 */
			
			IJ.log("Adding to ROI manager.");
			
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

            // Display results
            imp.setOverlay(overlay);
            imp.updateAndDraw();
            
            /*
             * Build a table to store bboxes.
             */
            
            IJ.log( "Generating result table." );

            // Show results table
            DetectionTableDialog tableDialog = new DetectionTableDialog();
            tableDialog.show(bboxes);
		
		}
		catch ( Exception e )
		{
			IJ.error( e.toString() );
		}
	}
	
	private ImagePlus preprocess(ImagePlus imp) {
		// TODO: update for movie
		
	    if (imp.getType() != ImagePlus.COLOR_RGB) {
	    	
	        return imp;
	    }

	    IJ.log( "Image has type ColorRGB, convert to 3-channels composite image." );
	    
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
}