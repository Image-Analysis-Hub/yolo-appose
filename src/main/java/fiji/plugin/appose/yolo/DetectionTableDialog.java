package fiji.plugin.appose.yolo;
import ij.IJ;
import ij.plugin.frame.RoiManager;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class DetectionTableDialog {

    private JFrame frame;
    private JTable table;
    private DefaultTableModel tableModel;

    // Column definitions
    private static final String[] COLUMNS = {
        "Index", "Class ID", "Class Name", "X", "Y", 
        "Width", "Height", "Confidence", "Area"
    };

    /**
     * Build and display the table from SAHI/YOLO results.
     */
    public void show(List<Map<String, Object>> bboxes) {

        // 1. Create table model
        tableModel = new DefaultTableModel(COLUMNS, 0) {
            @Override
            public boolean isCellEditable(int row, int col) {
                return false;  // read-only
            }

            @Override
            public Class<?> getColumnClass(int col) {
                switch (col) {
                    case 0: case 1: return Integer.class;
                    case 2: return String.class;
                    default: return Double.class;
                }
            }
        };

        // 2. Populate rows
        for (int i = 0; i < bboxes.size(); i++) {
            Map<String, Object> bbox = bboxes.get(i);

            double x      = ((Number) bbox.get("x")).doubleValue();
            double y      = ((Number) bbox.get("y")).doubleValue();
            double w      = ((Number) bbox.get("width")).doubleValue();
            double h      = ((Number) bbox.get("height")).doubleValue();
            double conf   = ((Number) bbox.get("confidence")).doubleValue();
            int classId   = ((Number) bbox.get("class_id")).intValue();
            String name   = (String) bbox.get("class_name");

            tableModel.addRow(new Object[]{
                i + 1, classId, name, x, y, w, h, conf, w * h
            });
        }

        // 3. Create JTable
        table = new JTable(tableModel);
        table.setAutoCreateRowSorter(true);        // clickable column sort
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.setFont(new Font("SansSerif", Font.PLAIN, 12));
        table.getTableHeader().setFont(new Font("SansSerif", Font.BOLD, 12));
        
        // Add after creating the JTable (Option 2)
        table.getSelectionModel().addListSelectionListener(e -> {
            if (e.getValueIsAdjusting()) return;

            int selectedRow = table.getSelectedRow();
            if (selectedRow < 0) return;

            // Convert view row to model row (in case table is sorted)
            int modelRow = table.convertRowIndexToModel(selectedRow);

            // Highlight corresponding ROI
            RoiManager rm = RoiManager.getInstance();
            if (rm != null) {
                rm.select(modelRow);
            }
        });

        // 4. Layout
        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setPreferredSize(new Dimension(750, 400));

        // 5. Buttons panel
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));

        JButton saveButton = new JButton("Save CSV...");
        saveButton.addActionListener(e -> saveAsCSV());

        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e -> frame.dispose());

        buttonPanel.add(saveButton);
        buttonPanel.add(closeButton);

        // 6. Summary label
        JLabel summary = new JLabel(String.format(
            "  Total detections: %d", bboxes.size()));
        summary.setFont(new Font("SansSerif", Font.ITALIC, 12));

        // 7. Assemble frame
        frame = new JFrame("YOLO Detection Results");
        frame.setLayout(new BorderLayout(5, 5));
        frame.add(summary, BorderLayout.NORTH);
        frame.add(scrollPane, BorderLayout.CENTER);
        frame.add(buttonPanel, BorderLayout.SOUTH);
        frame.pack();
        frame.setLocationRelativeTo(null);  // center on screen
        frame.setVisible(true);
    }

    /**
     * Save table contents to CSV via file chooser.
     */
    private void saveAsCSV() {
        JFileChooser fc = new JFileChooser();
        fc.setDialogTitle("Save Detections as CSV");
        fc.setSelectedFile(new java.io.File("detections.csv"));
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
            "CSV files (*.csv)", "csv"));

        int result = fc.showSaveDialog(frame);
        if (result != JFileChooser.APPROVE_OPTION) return;

        java.io.File file = fc.getSelectedFile();
        // Ensure .csv extension
        if (!file.getName().toLowerCase().endsWith(".csv")) {
            file = new java.io.File(file.getAbsolutePath() + ".csv");
        }

        try (FileWriter writer = new FileWriter(file)) {

            // Header
            for (int c = 0; c < tableModel.getColumnCount(); c++) {
                if (c > 0) writer.write(",");
                writer.write(tableModel.getColumnName(c));
            }
            writer.write("\n");

            // Rows
            for (int r = 0; r < tableModel.getRowCount(); r++) {
                for (int c = 0; c < tableModel.getColumnCount(); c++) {
                    if (c > 0) writer.write(",");
                    Object val = tableModel.getValueAt(r, c);
                    if (val instanceof Double) {
                        writer.write(String.format("%.4f", (Double) val));
                    } else {
                        writer.write(String.valueOf(val));
                    }
                }
                writer.write("\n");
            }

            IJ.log("Saved: " + file.getAbsolutePath());
            JOptionPane.showMessageDialog(frame,
                "Saved " + tableModel.getRowCount() + " detections to:\n" 
                + file.getAbsolutePath(),
                "Save Complete", JOptionPane.INFORMATION_MESSAGE);

        } catch (IOException ex) {
            IJ.error("Save Failed", ex.getMessage());
        }
    }
}
