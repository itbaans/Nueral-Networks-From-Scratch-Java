package Simple_NN;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MNIST_TRAINING {
    private static final int BATCH_SIZE = 1;  // You can adjust this value
    private static final Random random = new Random();
    NueralNetwork network;

    public MNIST_TRAINING() {
        // Initialize network with learning rate
        this.network = new NueralNetwork(0.05);
        
        // Set up network architecture
        Layer_NN inp = new Layer_NN(784, true, false);
        Layer_NN hidden1 = new Layer_NN(100, false, false);
        Layer_NN output = new Layer_NN(10, false, true);

        network.append(inp);
        network.append(hidden1);
        network.append(output);
    }

    public void train(String trainDataPath, String trainLabelsPath, int epochs) throws IOException {
        double[][] inputData = readAndTransposeCSV(trainDataPath, false);
        double[][] targetData = readAndTransposeCSV(trainLabelsPath, false);
        
        int numSamples = inputData[0].length;
        int numBatches = (numSamples + BATCH_SIZE - 1) / BATCH_SIZE;  // Ceiling division
        
        System.out.println("Training Progress:");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochError = 0;
            
            // Shuffle indices for each epoch
            int[] indices = generateShuffledIndices(numSamples);
            
            // Process each batch
            for (int batch = 0; batch < numBatches; batch++) {
                int startIdx = batch * BATCH_SIZE;
                int endIdx = Math.min(startIdx + BATCH_SIZE, numSamples);
                
                // Process each sample in the batch
                for (int i = startIdx; i < endIdx; i++) {
                    int idx = indices[i];
                    network.setInputLayer(getCol(inputData, idx));
                    network.train2(getCol(targetData, idx));
                }
                
                // Calculate batch error
                double[][] currentOutput = network.getCurrentOutput();
                double batchError = calculateError(currentOutput, 
                    extractColumns(targetData, indices, startIdx, endIdx));
                epochError += batchError;
            }
            
            // Print progress every 10 epochs
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                System.out.println("\nEpoch " + epoch + ":");
                System.err.println("Average Error: " + (epochError / numBatches));
                System.out.println();
            }
        }
    }

    public void test(String testDataPath, String testLabelsPath) throws IOException {
        double[][] testData = readAndTransposeCSV(testDataPath, false);
        double[][] testLabels = readAndTransposeCSV(testLabelsPath, false);
        
        int numCorrect = 0;
        int totalSamples = testData[0].length;
        
        for (int i = 0; i < totalSamples; i++) {
            double[][] input = getCol(testData, i);
            network.query(input);
            
            int predictedClass = getPredictedClass(network.getCurrentOutput());
            int actualClass = getActualClass(getCol(testLabels, i));
            
            if (predictedClass == actualClass) {
                numCorrect++;
            }
        }
        
        double accuracy = (double) numCorrect / totalSamples * 100;
        System.out.printf("Test Accuracy: %.2f%% (%d/%d)%n", 
            accuracy, numCorrect, totalSamples);
    }

    private int[] generateShuffledIndices(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        
        // Fisher-Yates shuffle
        for (int i = size - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }

    private double[][] extractColumns(double[][] data, int[] indices, 
                                    int startIdx, int endIdx) {
        double[][] result = new double[data.length][endIdx - startIdx];
        for (int i = startIdx; i < endIdx; i++) {
            for (int j = 0; j < data.length; j++) {
                result[j][i - startIdx] = data[j][indices[i]];
            }
        }
        return result;
    }

    private int getPredictedClass(double[][] output) {
        int maxIdx = 0;
        double maxVal = output[0][0];
        for (int i = 1; i < output.length; i++) {
            if (output[i][0] > maxVal) {
                maxVal = output[i][0];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private int getActualClass(double[][] target) {
        for (int i = 0; i < target.length; i++) {
            if (target[i][0] == 0.99) {
                return i;
            }
        }
        return -1;  // Error case
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

    public static double[][] readAndTransposeCSV(String filePath, boolean skipHeader) 
            throws IOException {
        // Existing CSV reading code remains the same
        List<String[]> csvData = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            if (skipHeader) {
                br.readLine();
            }
            
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                csvData.add(values);
            }
        }
        
        int rows = csvData.get(0).length;
        int cols = csvData.size();
        double[][] transposed = new double[rows][cols];
        
        for (int i = 0; i < cols; i++) {
            String[] currentRow = csvData.get(i);
            for (int j = 0; j < rows; j++) {
                try {
                    transposed[j][i] = Double.parseDouble(currentRow[j].trim());
                } catch (NumberFormatException e) {
                    System.err.println("Warning: Could not parse value at row " + i + 
                        ", column " + j);
                    transposed[j][i] = 0.0;
                }
            }
        }
        
        return transposed;
    }

    public static void main(String[] args) throws Exception {
        MNIST_TRAINING trainer = new MNIST_TRAINING();
        
        // Training phase
        trainer.train("TheVanilla\\rescaled_dataset_train.csv",
                     "TheVanilla\\encoded_labels_train.csv",
                     15);  // 15 epochs
        
        // Testing phase
        trainer.test("TheVanilla\\rescaled_dataset_test.csv",
                    "TheVanilla\\encoded_labels_test.csv");
    }
}