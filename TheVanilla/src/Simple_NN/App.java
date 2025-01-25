package Simple_NN;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class App {
    public static void main(String[] args) throws Exception {
        // Increased learning rate
        NueralNetwork network = new NueralNetwork(0.1);
        
        double[][] input = readAndTransposeCSV("TheVanilla\\rescaled_dataset_train.csv", false);
        double[][] target = readAndTransposeCSV("TheVanilla\\encoded_labels_train.csv", false);


        // Modified architecture
        Layer_NN inp = new Layer_NN(784, true, false);
        Layer_NN hidden1 = new Layer_NN(100, false, false);
        //Layer hidden2 = new Layer(64, false, false);
        Layer_NN output = new Layer_NN(10, false, true);

        network.append(inp);
        network.append(hidden1);
        //network.append(hidden2);
        network.append(output);
        
        System.out.println("Training Progress:");
        int limit = 15;
        for(int epoch = 0; epoch < limit; epoch++) {
            //if(epoch % 10 == 0) network.checkLayerActivations();
            for(int i = 0; i < input[0].length; i++) {
                network.setInputLayer(getCol(input, i));
                network.train2(getCol(target, i));
            }

            if(epoch % 10 == 0) {
                System.out.println("\nEpoch " + epoch + ":");
                System.err.println(calculateError(network.getCurrentOutput(), target));
                System.out.println();
            }

            if(epoch == limit - 1) {
                //network.checkLayerActivations();
                System.out.println("\nEpoch " + epoch + ":");
                System.err.println(calculateError(network.getCurrentOutput(), target));
                System.out.println();
            }
        }

        double[][] test = readAndTransposeCSV("TheVanilla\\rescaled_dataset_test.csv", false);
        
        double[][] entry = new double[784][1];
        for (int i = 0; i < 784; i++) {
            entry[i][0] = test[i][0];
        }
        network.query(entry);
        for (int i = 0; i < 784; i++) {
            entry[i][0] = test[i][3];
        }
        System.out.println("______________");
        network.query(entry);
        //network.printLayerInfo("ff");


        double[][] labels = readAndTransposeCSV("TheVanilla\\test.csv", false);
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                System.out.print(labels[i][j]+" ");
            }
            System.out.println();
        }
    }

    private static double[][] getCol(double[][] t, int c) {
        double[][] r = new double[t.length][1];
        for (int i = 0; i < t.length; i++) {
            r[i][0] = t[i][c];
        }
        return r;
    }

    private static double calculateError(double[][] output, double[][] target) {
        double error = 0;
        for(int i = 0; i < output.length; i++) {
            for(int j = 0; j < output[0].length; j++) {
                error += Math.pow(target[i][j] - output[i][j], 2);
            }
        }
        return error / (output.length * output[0].length);
    }

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
}