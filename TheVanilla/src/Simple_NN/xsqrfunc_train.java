package Simple_NN;

import java.util.ArrayList;
import java.util.List;

import java.io.PrintWriter;

public class xsqrfunc_train {

    //MAKE DATA SET OF Y=X^2 FUNCTION
    public static void main(String[] args) {

        // Create a dataset of y = x^2
        double[][] x = new double[1000][1];
        double[][] y = new double[1000][1];

        //X -> (-1000, 1000)
        for (int i = 0; i < 1000; i++) {
            // Scale x values to [-1, 1]
            x[i][0] = (i - 500) / 500.0;
            // Calculate y values as x^2
            y[i][0] = x[i][0] * x[i][0];
        }

        // Create a neural network
        NueralNetwork nn = new NueralNetwork(0.05);

        //SIGMOID = 0;
        //RELU = 1;
        //TANH = 2;
        //LEAKY_RELU = 3;

        // Add an input layer with 1 neuron
        nn.append(new Layer_NN(1, true, false));

        // Add hidden layers
        nn.append(new Layer_NN(25, false, false, 3));

        //nn.append(new Layer_NN(50, false, false, 3));

        nn.append(new Layer_NN(25, false, false, 3));

        // Add an output layer with 1 neuron and a sigmoid activation function
        nn.append(new Layer_NN(1, false, true, 3));

        //train for 1000 epochs
        double[][] x_copy = new double[1000][1];
        double[][] y_copy = new double[1000][1];

        for (int i = 0; i < 1000; i++) {
            x_copy[i][0] = x[i][0];
            y_copy[i][0] = y[i][0];
        }

        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 1000; j++) {
                nn.setInputLayer(new double[][]{{x_copy[j][0]}});
                nn.train2(new double[][]{{y_copy[j][0]}});
            }
            //shuffle data for better training
            // for (int j = 0; j < 1000; j++) {
            //     int a = (int) (Math.random() * 1000);
            //     int b = (int) (Math.random() * 1000);
            //     double temp = x[a][0];
            //     x_copy[a][0] = x_copy[b][0];
            //     x_copy[b][0] = temp;
            //     temp = y_copy[a][0];
            //     y_copy[a][0] = y_copy[b][0];
            //     y_copy[b][0] = temp;
            // }
        }

        double[][] x_test = new double[2000][1];
        double[][] y_test = new double[2000][1];

        for (int i = 0; i < 2000; i++) {
            // Scale x values to [-1, 1]
            x_test[i][0] = (i - 1000) / 1000.0;
            // Calculate y values as x^2
            y_test[i][0] = x_test[i][0] * x_test[i][0];
        }

        List<Double> xValues = new ArrayList<>();
        List<Double> trueY = new ArrayList<>();
        List<Double> predictedY = new ArrayList<>();


        for (int i = 0; i < 2000; i++) {
            nn.setInputLayer(new double[][]{{x_test[i][0]}});

            System.out.println("For X: " + x_test[i][0]);
            System.out.println("True Value: " + y_test[i][0]);

            nn.query(new double[][]{{y_test[i][0]}});

            for (double[] val : nn.getCurrentOutput()) {
                System.out.println("Predicted Value: " + val[0]);
                // Collect data for plotting
                xValues.add(x_test[i][0]);
                trueY.add(y_test[i][0]);
                predictedY.add(val[0]);
            }
        }

        try (PrintWriter writer = new PrintWriter("TheVanilla\\src\\Simple_NN\\plot_data.csv")) {
            writer.println("x,true_y,predicted_y");
            for (int i = 0; i < xValues.size(); i++) {
                writer.println(xValues.get(i) + "," + trueY.get(i) + "," + predictedY.get(i));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }


    }
    
}
