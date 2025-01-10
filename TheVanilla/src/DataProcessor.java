import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataProcessor {
    /**
     * Reads a CSV file and converts it to a double[][] array with transposition
     * @param filePath Path to the CSV file
     * @param skipHeader Whether to skip the first line (header)
     * @return Transposed double[][] array
     */
    public static double[][] readAndTransposeCSV(String filePath, boolean skipHeader) throws IOException {
        List<String[]> csvData = new ArrayList<>();
        
        // Read CSV file
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            if (skipHeader) {
                br.readLine(); // Skip header line
            }
            
            while ((line = br.readLine()) != null) {
                // Split by comma, handling possible quoted values
                String[] values = line.split(",");
                csvData.add(values);
            }
        }
        
        // Get dimensions for the transposed array
        int rows = csvData.get(0).length;    // Original columns become rows
        int cols = csvData.size();           // Original rows become columns
        
        // Create the transposed array
        double[][] transposed = new double[rows][cols];
        
        // Fill the transposed array
        for (int i = 0; i < cols; i++) {
            String[] currentRow = csvData.get(i);
            for (int j = 0; j < rows; j++) {
                try {
                    transposed[j][i] = Double.parseDouble(currentRow[j].trim());
                } catch (NumberFormatException e) {
                    System.err.println("Warning: Could not parse value at row " + i + ", column " + j);
                    transposed[j][i] = 0.0; // Default value for unparseable entries
                }
            }
        }
        
        return transposed;
    }
    
    /**
     * Reads a labels CSV file and converts it to a double[][] array suitable for training
     * @param filePath Path to the labels CSV file
     * @param skipHeader Whether to skip the first line (header)
     * @return Processed double[][] array of labels
     */
    public static double[][] readLabelsCSV(String filePath, boolean skipHeader) throws IOException {
        List<String> labels = new ArrayList<>();
        
        // Read CSV file
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            if (skipHeader) {
                br.readLine(); // Skip header line
            }
            
            while ((line = br.readLine()) != null) {
                labels.add(line.trim());
            }
        }
        
        // Create the labels array with one row
        double[][] labelArray = new double[1][labels.size()];
        
        // Fill the array
        for (int i = 0; i < labels.size(); i++) {
            try {
                labelArray[0][i] = Double.parseDouble(labels.get(i));
            } catch (NumberFormatException e) {
                System.err.println("Warning: Could not parse label at position " + i);
                labelArray[0][i] = 0.0; // Default value for unparseable entries
            }
        }
        
        return labelArray;
    }
    
    // /**
    //  * Helper method to print array contents for verification
    //  * @param arr Array to print
    //  * @param label Label to print before array
    //  */
    // public static void printArray(double[][] arr, String label) {
    //     System.out.println(label + ":");
    //     System.out.println("Dimensions: " + arr.length + " x " + arr[0].length);
        
    //     // Print first few elements as sample
    //     int maxRows = Math.min(5, arr.length);
    //     int maxCols = Math.min(5, arr[0].length);
        
    //     for (int i = 0; i < maxRows; i++) {
    //         for (int j = 0; j < maxCols; j++) {
    //             System.out.printf("%.2f\t", arr[i][j]);
    //         }
    //         System.out.println(j < arr[0].length ? "..." : "");
    //     }
    //     if (arr.length > maxRows) {
    //         System.out.println("...");
    //     }
    // }
}