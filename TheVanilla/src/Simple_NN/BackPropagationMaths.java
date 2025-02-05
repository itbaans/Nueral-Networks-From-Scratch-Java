package Simple_NN;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.RealMatrix;

public class BackPropagationMaths {
    
    public static RealMatrix mse_der(RealMatrix preds, RealMatrix targets) {

        int batchSize = preds.getColumnDimension(); // Assuming column dimension represents the batch size
        return targets.subtract(preds).scalarMultiply(2.0 / batchSize);

    }

    public static RealMatrix sigmoid_der(RealMatrix activations) {

        RealMatrix b = activations.scalarMultiply(-1).scalarAdd(1.0);
        return elementWiseMult(activations, b);

    }

    //make function for leaky_relu derivative
    public static RealMatrix leaky_relu_der(RealMatrix activations) {

        RealMatrix m = new Array2DRowRealMatrix(new double[activations.getRowDimension()][activations.getColumnDimension()]);

        for (int i = 0; i < activations.getRowDimension(); i++) {

            for (int j = 0; j < activations.getColumnDimension(); j++) {

                if (activations.getEntry(i, j) <= 0) m.setEntry(i, j, 0.01);
                else m.setEntry(i, j, 1.0);

            }

        }

        return m;

    }

    public static RealMatrix tanh_der(RealMatrix activations) {

        RealMatrix b = activations.copy();
        return b.scalarMultiply(-1).scalarAdd(1).scalarMultiply(1).scalarMultiply(1).subtract(elementWiseMult(b, b));

    }

    public static RealMatrix reLu_der(RealMatrix activations) {

        RealMatrix m = new Array2DRowRealMatrix(new double[activations.getRowDimension()][activations.getColumnDimension()]);

        for (int i = 0; i < activations.getRowDimension(); i++) {

            for (int j = 0; j < activations.getColumnDimension(); j++) {

                if (activations.getEntry(i, j) <= 0) m.setEntry(i, j, 0);
                else m.setEntry(i, j, 1.0);

            }

        }

        return m;

    }


    public static RealMatrix get_derivative(int act_type, RealMatrix activations) {

        switch(act_type) {

            case 0:
                return sigmoid_der(activations);

            case 1:
                return reLu_der(activations);

            case 2:
                return tanh_der(activations);

            case 3:
                return leaky_relu_der(activations);

            default:
                return sigmoid_der(activations);

        }

    }

    public static RealMatrix elementWiseMult(RealMatrix a, RealMatrix b) {

        RealMatrix m = new Array2DRowRealMatrix(new double[a.getRowDimension()][a.getColumnDimension()]);

        for (int i = 0; i < a.getRowDimension(); i++) {

            for (int j = 0; j < a.getColumnDimension(); j++) {

                m.setEntry(i, j, a.getEntry(i, j) * b.getEntry(i, j));

            }

        }

        return m;

    }

}
