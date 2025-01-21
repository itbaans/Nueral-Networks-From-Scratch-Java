package Simple_NN;

import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import java.util.Random;

public class Layer_NN extends Layer {

    RealMatrix nodes;
    RealMatrix weights;
    RealMatrix errors;
    Layer_NN nextLayer;
    Layer_NN prevLayer;
    boolean isInput;
    boolean isOutput;
    Random rand = new Random();

    public Layer_NN(int no_of_nodes, boolean in, boolean out) {

        isInput = in;
        isOutput = out;
        if(isInput) nodes = initializeRandomMatrix(no_of_nodes, 1, true);
        else nodes = MatrixUtils.createRealMatrix(new double[no_of_nodes][1]);
                
    }

    public void setInputs(double[][] inputs) {

        if (isInput) nodes = MatrixUtils.createRealMatrix(inputs);
        else System.out.println("Not an input layer");

    }

    public void connect_next(Layer l) {

        if(!isOutput) {

            nextLayer = (Layer_NN) l;
            weights = initializeRandomMatrix(nextLayer.nodes.getRowDimension(), nodes.getRowDimension(), true);

            evaluateNextNodes();

        }

        else System.out.println("cannot connect output layer");

    }

    public void evaluateNextNodes() {

        if(weights != null) {

            nextLayer.nodes = weights.multiply(nodes);
            for (int i = 0; i < nextLayer.nodes.getRowDimension(); i++) {
                for (int j = 0; j < nextLayer.nodes.getColumnDimension(); j++) {
                    nextLayer.nodes.setEntry(i, j, Activations.sigmoid(nextLayer.nodes.getEntry(i, j)));
                }
                
            }

        }

    }
    
    public void printNodesValues() {

        for (int i = 0; i < nodes.getRowDimension(); i++) {
            for (int j = 0; j < nodes.getColumnDimension(); j++) {
                System.out.print(nodes.getEntry(i, j)+" ");
            }          
            System.out.println();
        }

    }

    public void printErrorValues() {

        for (int i = 0; i < errors.getRowDimension(); i++) {
            for (int j = 0; j < errors.getColumnDimension(); j++) {
                System.out.print(Math.round(nodes.getEntry(i, j))+" ");
            } 
            System.out.println();         
        }

    }

    public void connect_prev(Layer l) {
        prevLayer = (Layer_NN) l;
    }

    public RealMatrix initializeRandomMatrix(int row, int col, boolean useGaussian) {
        double[][] t = new double[row][col];
        // Xavier/Glorot initialization
        double stdDev = Math.sqrt(2.0 / (row + col));
        
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                if (useGaussian) {
                    // Limit the range to prevent extreme values
                    double value = rand.nextGaussian() * stdDev;
                    // Clip weights to prevent extreme values
                    //System.out.println("VAL TAST: "+value);
                    value = Math.max(-1.0, Math.min(1.0, value));
                    t[r][c] = value;
                } else {
                    // Uniform initialization between -sqrt(6)/sqrt(n+m) and sqrt(6)/sqrt(n+m)
                    //double limit = Math.sqrt(6.0 / (row + col));
                    t[r][c] = (rand.nextDouble() * 4 - 2);
                }
            }
        }
        return MatrixUtils.createRealMatrix(t);
    }
    
    
}
