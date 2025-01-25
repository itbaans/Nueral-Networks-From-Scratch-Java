package Simple_NN;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

public class NueralNetwork {
    
    Layer_NN inputLayer;
    Layer_NN outputLayer;
    double learningRate;

    public NueralNetwork(double lr) {

        learningRate = lr;
        
    }

    public void setInputLayer(double [][] inp) {

        Layer_NN l = new Layer_NN(inp[0].length, true, false);
        l.setInputs(inp);

        if(inputLayer == null) inputLayer = l;
        else {
            Layer_NN t = inputLayer.nextLayer;
            l.weights = inputLayer.weights;
            inputLayer = l;
            l.nextLayer = t;
            t.prevLayer = inputLayer;
        }
    } 

    public void append(Layer_NN l) {

        if(inputLayer == null) {

            if(!l.isInput) {
                System.out.println("first layer must be input");
            }

            inputLayer = l;
            return;

        }

        Layer_NN currLayer = inputLayer;

        while(currLayer.nextLayer != null) {

            currLayer = currLayer.nextLayer;

        }

        currLayer.connect_next((Layer)l);
        l.connect_prev((Layer)currLayer);
        outputLayer = l;

    }

    public void checkLayerActivations() {
        Layer_NN curr = inputLayer;
        int layerNum = 0;
        while(curr != null) {
            double sum = 0;
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            
            // Calculate statistics for this layer's nodes
            for(int i = 0; i < curr.nodes.getRowDimension(); i++) {
                double val = curr.nodes.getEntry(i, 0);
                sum += val;
                min = Math.min(min, val);
                max = Math.max(max, val);
            }
            double avg = sum / curr.nodes.getRowDimension();
            
            System.out.printf("Layer %d stats: avg=%.6f, min=%.6f, max=%.6f%n", 
                            layerNum, avg, min, max);
            
            if(curr.weights != null) {
                // Also check weights
                double wSum = 0;
                double wMin = Double.MAX_VALUE;
                double wMax = Double.MIN_VALUE;
                
                for(int i = 0; i < curr.weights.getRowDimension(); i++) {
                    for(int j = 0; j < curr.weights.getColumnDimension(); j++) {
                        double w = curr.weights.getEntry(i, j);
                        wSum += w;
                        wMin = Math.min(wMin, w);
                        wMax = Math.max(wMax, w);
                    }
                }
                double wAvg = wSum / (curr.weights.getRowDimension() * 
                                    curr.weights.getColumnDimension());
                
                System.out.printf("Layer %d weights: avg=%.6f, min=%.6f, max=%.6f%n", 
                                layerNum, wAvg, wMin, wMax);
            }
            
            curr = curr.nextLayer;
            layerNum++;
        }
    }

    public void train2(double[][] tars) {

        RealMatrix targets = MatrixUtils.createRealMatrix(tars);
    
        // Forward pass
        flowTheLayerss();  

        //final layer error
        RealMatrix mse_der = BackPropagationMaths.mse_der(outputLayer.nodes, targets);
        RealMatrix act_der = BackPropagationMaths.get_derivative(outputLayer.act_type, outputLayer.nodes);

        outputLayer.errors = elementWiseMult(mse_der, act_der);

        // Backpropagation
        Layer_NN curr = outputLayer;

        while(curr != inputLayer) {
            // Propagate errors backward
            RealMatrix t1 = curr.prevLayer.weights.transpose().multiply(curr.errors);
            RealMatrix t2 = BackPropagationMaths.get_derivative(curr.prevLayer.act_type, curr.prevLayer.nodes);
            curr.prevLayer.errors = elementWiseMult(t1, t2);
            
            // Calculate weight updates using the chain rule
            RealMatrix gradients = curr.errors.multiply(curr.prevLayer.nodes.transpose());

            // Update weights using gradient descent
            curr.prevLayer.weights = curr.prevLayer.weights.add(
                gradients.scalarMultiply(learningRate / curr.errors.getColumnDimension())
            );

            curr = curr.prevLayer;
        }

    }

    public void train(double[][] tars) {
        RealMatrix targets = MatrixUtils.createRealMatrix(tars);
    
        // Forward pass
        flowTheLayerss();
        
        // Calculate initial error at output layer
        outputLayer.errors = targets.subtract(outputLayer.nodes);
    
        // Backpropagation
        Layer_NN curr = outputLayer;
        while(curr != inputLayer) {
            // Propagate errors backward
            curr.prevLayer.errors = curr.prevLayer.weights.transpose().multiply(curr.errors);
            
            // Calculate weight updates using the chain rule
            RealMatrix a = elementWiseMult(curr.errors, curr.nodes);
            RealMatrix b = elementWiseMult(a, curr.nodes.scalarMultiply(-1).scalarAdd(1.0));
            RealMatrix gradients = b.multiply(curr.prevLayer.nodes.transpose());

            // Update weights using gradient descent
            curr.prevLayer.weights = curr.prevLayer.weights.add(
                gradients.scalarMultiply(learningRate / a.getColumnDimension())
            );

            curr = curr.prevLayer;
        }
    }

    private RealMatrix clipGradients(RealMatrix gradients, double clipValue) {
        // Iterate through each element in the matrix and clip it
        double[][] data = gradients.getData(); // Extract the matrix as a 2D array
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = Math.max(-clipValue, Math.min(clipValue, data[i][j]));
            }
        }
        return MatrixUtils.createRealMatrix(data); // Recreate the matrix with clipped values
    }

    private void debugAvg(RealMatrix t) {
        double sum = 0;
        for (int i = 0; i < t.getRowDimension(); i++) {
            for (int j = 0; j < t.getColumnDimension(); j++) {
                System.out.println(t.getEntry(i, j));
                //sum+=t.getEntry(i, j);
            }
        }
        // System.out.println(sum);
        // System.out.println(sum/(t.getRowDimension() * t.getColumnDimension()));
    }

    public RealMatrix elementWiseMult(RealMatrix a, RealMatrix b) {

        RealMatrix m = new Array2DRowRealMatrix(new double[a.getRowDimension()][a.getColumnDimension()]);

        for (int i = 0; i < a.getRowDimension(); i++) {

            for (int j = 0; j < a.getColumnDimension(); j++) {

                m.setEntry(i, j, a.getEntry(i, j) * b.getEntry(i, j));

            }

        }

        return m;

    }

    public void printOutPutErrors() {
        outputLayer.printErrorValues();
    }

    public void query(double[][] inps) {

        inputLayer.setInputs(inps);
        flowTheLayerss();
        //outputLayer.printNodesValues();

    }

    public void flowTheLayerss() {

        Layer_NN curr = inputLayer;

        while(curr != outputLayer) {

            curr.evaluateNextNodes();
            curr = curr.nextLayer;

        }
    }


    public void printLayerNodes() {

        Layer_NN curr = inputLayer;

        while(curr != null) {

            curr.printNodesValues();
            System.out.println("---------------");
            curr = curr.nextLayer;

        }


    }

    public void printLayerInfo(String label) {
        System.out.println(label + ":");
        Layer_NN curr = inputLayer;
        int layerNum = 0;
        while(curr != null) {
            System.out.println("Layer " + layerNum + " values:");
            curr.printNodesValues();
            if(curr.weights != null) {
                System.out.println("Weights to next layer:");
                for(int i = 0; i < curr.weights.getRowDimension(); i++) {
                    for(int j = 0; j < curr.weights.getColumnDimension(); j++) {
                        System.out.printf("%.4f ", curr.weights.getEntry(i, j));
                    }
                    System.out.println();
                }
            }
            curr = curr.nextLayer;
            layerNum++;
        }
    }

    public double[][] getCurrentOutput() {
        
        int r = outputLayer.nodes.getRowDimension();
        int c = outputLayer.nodes.getColumnDimension();
        double[][] vals = new double[r][c];

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                vals[i][j] = outputLayer.nodes.getEntry(i, j);
            }
        }
        
        return vals;
    }

}


